#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import argparse
import numpy as np
import mxnet as mx
import cv2
import time
from queue import Queue

from generate_anchor import generate_anchors_fpn, nonlinear_pred, generate_runtime_anchors
from numpy import frombuffer, uint8, concatenate, float32, block, maximum, minimum
from mxnet.ndarray import waitall, array
from functools import partial

from threading import Thread


class BaseDetection:
    def __init__(self, *, thd, gpu, margin, nms_thd, verbose):
        self.threshold = thd
        self.nms_threshold = nms_thd
        self.device = gpu
        self.margin = margin

        self._queue = Queue(4)
        self.write_queue = self._queue.put_nowait
        self.read_queue = iter(self._queue.get, b'')

        self._nms_wrapper = partial(
            self.non_maximum_suppression, threshold=self.nms_threshold)

    def margin_clip(self, b):
        margin_x = (b[2] - b[0]) * self.margin
        margin_y = (b[3] - b[1]) * self.margin

        b[0] -= margin_x
        b[1] -= margin_y
        b[2] += margin_x
        b[3] += margin_y

        return np.clip(b, 0, None, out=b)

    @staticmethod
    def non_maximum_suppression(dets, threshold):
        ''' ##### Author 1996scarlet@gmail.com
        merged_non_maximum_suppression
        Greedily select boxes with high confidence and overlap with threshold.
        If the boxes' overlap > threshold, we consider they are the same one.

        Parameters
        ----------
        dets: ndarray
            Bounding boxes of shape [N, 5].
            Each box has [x1, y1, x2, y2, score].

        threshold: float
            The src scales para.

        Returns
        -------
        Generator of kept box, each box has [x1, y1, x2, y2, score].

        Usage
        -----
        >>> for res in non_maximum_suppression(dets, thresh):
        >>>     pass
        '''

        x1, y1, x2, y2, scores = dets.T

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        while order.size > 0:
            keep, others = order[0], order[1:]

            yield dets[keep]

            xx1 = maximum(x1[keep], x1[others])
            yy1 = maximum(y1[keep], y1[others])
            xx2 = minimum(x2[keep], x2[others])
            yy2 = minimum(y2[keep], y2[others])

            w = maximum(0.0, xx2 - xx1 + 1)
            h = maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            overlap = inter / (areas[keep] - inter + areas[others])

            order = others[overlap < threshold]

    def non_maximum_selection(self, x):
        return x[:1]

    # def detect(self, src, **kwargs):
    #     raise NotImplementedError('Not Implemented Function: detect')

    @staticmethod
    def filter_boxes(boxes, min_size, max_size=-1):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        if max_size > 0:
            boxes = np.where(minimum(ws, hs) < max_size)[0]
        if min_size > 0:
            boxes = np.where(maximum(ws, hs) > min_size)[0]
        return boxes


class MxnetDetectionModel(BaseDetection):
    def __init__(self, prefix, epoch, scale, gpu=-1, thd=0.6, margin=0,
                 nms_thd=0.4, verbose=False):

        super().__init__(thd=thd, gpu=gpu, margin=margin, nms_thd=nms_thd, verbose=verbose)

        self.scale = scale
        self._rescale = partial(cv2.resize, dsize=None, fx=self.scale,
                                fy=self.scale, interpolation=cv2.INTER_LINEAR)

        self._ctx = mx.cpu() if self.device < 0 else mx.gpu(self.device)
        self._fpn_anchors = generate_anchors_fpn().items()
        self._runtime_anchors = {}

        model = self._load_model(prefix, epoch)

        self._forward = partial(model.forward, is_train=False)
        self._solotion = model.get_outputs

    def _load_model(self, prefix, epoch):
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        model = mx.mod.Module(sym, context=self._ctx, label_names=None)
        model.bind(data_shapes=[('data', (1, 3, 640, 480))],
                   for_training=False)
        model.set_params(arg_params, aux_params)
        return model

    def _get_runtime_anchors(self, height, width, stride, base_anchors):
        key = height, width, stride
        if key not in self._runtime_anchors:
            self._runtime_anchors[key] = generate_runtime_anchors(
                height, width, stride, base_anchors).reshape((-1, 4))
        return self._runtime_anchors[key]

    def _retina_solving(self, out):
        """
        Parameters
        ----------
        height: height of plane
        width:  width of plane
        stride: stride ot the original image
        anchors_base: (A, 4) a base set of anchors

        Returns
        -------
        all_anchors: (height * width, A, 4) ndarray of anchors spreading over the plane
        """

        for fpn in self._fpn_anchors:
            scores, deltas = next(out)[:, fpn[1].shape[0]:, :, :], next(out)

            anchors = self._get_runtime_anchors(*deltas.shape[-2:], *fpn)

            scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
            deltas = deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

            mask = scores.reshape((-1,)) > self.threshold
            proposals = deltas[mask]

            nonlinear_pred(anchors[mask], proposals)

            yield [proposals / self.scale, scores[mask]]

    def _retina_detach(self, out):
        out = map(lambda x: x.asnumpy(), out)
        return block(list(self._retina_solving(out)))

    def _retina_forward(self, src):
        ''' ##### Author 1996scarlet@gmail.com
        Image preprocess and return the forward results.

        Parameters
        ----------
        src: ndarray
            The image batch of shape [H, W, C].

        scales: list of float
            The src scales para.

        Returns
        -------
        net_out: list, len = STEP * N
            If step is 2, each block has [scores, bbox_deltas]
            Else if step is 3, each block has [scores, bbox_deltas, landmarks]

        Usage
        -----
        >>> out = self._retina_forward(frame)
        '''

        # timea = time.perf_counter()
        dst = self._rescale(src)
        data = array(dst.transpose((2, 0, 1))[None, ...])
        db = mx.io.DataBatch(data=(data, ))
        self._forward(db)
        return self._solotion()
        # print(f'inferance: {time.perf_counter() - timeb}')

    def workflow_inference(self, instream):
        for source in instream:
            # st = time.perf_counter()

            frame = frombuffer(source, dtype=uint8).reshape(V_H, V_W, V_C)
            out = self._retina_forward(frame)

            try:
                self.write_queue((frame, out))
            except:
                waitall()
                print('Frame queue full', file=sys.stderr)

            # print(f'workflow_inference: {time.perf_counter() - st}')

    def workflow_postprocess(self, outstream=None):
        for frame, out in self.read_queue:
            # st = time.perf_counter()
            detach = self._retina_detach(out)
            # dets = self.non_maximum_selection(detach)  # 1.7 us
            # print(f'workflow_postprocess: {time.perf_counter() - st}')

            if outstream is None:
                for res in self._nms_wrapper(detach):
                    # self.margin_clip(res)

                    cv2.rectangle(frame, (res[0], res[1]),
                                  (res[2], res[3]), (255, 255, 0))

                cv2.imshow('res', frame)
                cv2.waitKey(1)
            else:
                outstream(frame)
                outstream(res)


if __name__ == '__main__':
    import sys

    V_W, V_H, V_C = 640, 480, 3
    BUFFER_SIZE = V_W * V_H * V_C

    read = sys.stdin.buffer.read
    write = sys.stdout.buffer.write
    camera = iter(partial(read, BUFFER_SIZE), b'')

    fd = MxnetDetectionModel("weights/16and32", 0,
                             scale=.4, gpu=0, margin=0.15)

    poster = Thread(target=fd.workflow_postprocess)
    poster.start()

    infer = Thread(target=fd.workflow_inference, args=(camera,))
    infer.daemon = True
    infer.start()
