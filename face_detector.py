#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import mxnet as mx
import cv2
import time
from queue import Queue, Full

from generate_anchor import generate_anchors_fpn, nonlinear_pred, generate_runtime_anchors
from numpy import frombuffer, uint8, concatenate, float32, block, maximum, minimum, prod
from mxnet.ndarray import waitall, concat
from functools import partial

from threading import Thread


class BaseDetection:
    def __init__(self, *, thd, gpu, margin, nms_thd, verbose):
        self.threshold = thd
        self.nms_threshold = nms_thd
        self.device = gpu
        self.margin = margin

        self._queue = Queue(200)
        self.write_queue = self._queue.put_nowait
        self.read_queue = iter(self._queue.get, b'')

        self._nms_wrapper = partial(self.non_maximum_suppression,
                                    threshold=self.nms_threshold)
        
        self._biggest_wrapper = partial(self.find_biggest_box)


    def margin_clip(self, b):
        margin_x = (b[2] - b[0]) * self.margin
        margin_y = (b[3] - b[1]) * self.margin

        b[0] -= margin_x
        b[1] -= margin_y
        b[2] += margin_x
        b[3] += margin_y

        return np.clip(b, 0, None, out=b)

    @staticmethod
    def find_biggest_box(dets):
        return max(dets, key=lambda x: x[4]) if dets.size > 0 else None
        # return max(dets, key=lambda x: x[0]) if dets.size > 0 else None

    @staticmethod
    def non_maximum_suppression(dets, threshold):
        ''' ##### Author 1996scarlet@gmail.com
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

            yield np.copy(dets[keep])

            xx1 = maximum(x1[keep], x1[others])
            yy1 = maximum(y1[keep], y1[others])
            xx2 = minimum(x2[keep], x2[others])
            yy2 = minimum(y2[keep], y2[others])

            w = maximum(0.0, xx2 - xx1 + 1)
            h = maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            overlap = inter / (areas[keep] - inter + areas[others])

            order = others[overlap < threshold]

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
    def __init__(self, prefix, epoch, scale=1., gpu=-1, thd=0.6, margin=0,
                 nms_thd=0.4, verbose=False):

        super().__init__(thd=thd, gpu=gpu, margin=margin,
                         nms_thd=nms_thd, verbose=verbose)

        self.scale = scale
        self._rescale = partial(cv2.resize, dsize=None, fx=self.scale,
                                fy=self.scale, interpolation=cv2.INTER_NEAREST)

        self._ctx = mx.cpu() if self.device < 0 else mx.gpu(self.device)
        self._fpn_anchors = generate_anchors_fpn()
        self._runtime_anchors = {}

        self.model = self._load_model(prefix, epoch)
        self.exec_group = self.model._exec_group

    def _load_model(self, prefix, epoch):
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        model = mx.mod.Module(sym, context=self._ctx, label_names=None)
        model.bind(data_shapes=[('data', (1, 3, 1, 1))],
                   for_training=False)
        model.set_params(arg_params, aux_params)
        return model

    def _get_runtime_anchors(self, height, width, stride, base_anchors):
        key = height, width, stride
        if key not in self._runtime_anchors:
            self._runtime_anchors[key] = generate_runtime_anchors(
                height, width, stride, base_anchors).reshape((-1, 4))
        return self._runtime_anchors[key]

    def _retina_detach(self, out):
        ''' ##### Author 1996scarlet@gmail.com
        Solving bounding boxes.

        Parameters
        ----------
        out: map object of staggered scores and deltas.
            scores, deltas = next(out), next(out)

            Each scores has shape [N, A*4, H, W].
            Each deltas has shape [N, A*4, H, W].

            N is the batch size.
            A is the shape[0] of base anchors declared in the fpn dict.
            H, W is the heights and widths of the anchors grid,
            based on the stride and input image's height and width.

        Returns
        -------
        Generator of list, each list has [boxes, scores].

        Usage
        -----
        >>> np.block(list(self._retina_solving(out)))
        '''

        buffer, anchors = out[0].asnumpy(), out[1]
        mask = buffer[:, 4] > self.threshold
        deltas = buffer[mask]
        nonlinear_pred(anchors[mask], deltas)
        deltas[:, :4] /= self.scale
        return deltas

    def _retina_solve(self):
        out, res, anchors = iter(self.exec_group.execs[0].outputs), [], []

        for fpn in self._fpn_anchors:
            scores = next(out)[:, -fpn.scales_shape:,
                               :, :].transpose((0, 2, 3, 1))
            deltas = next(out).transpose((0, 2, 3, 1))

            res.append(concat(deltas.reshape((-1, 4)),
                              scores.reshape((-1, 1)), dim=1))

            anchors.append(self._get_runtime_anchors(*deltas.shape[1:3],
                                                     fpn.stride,
                                                     fpn.base_anchors))

        return concat(*res, dim=0), concatenate(anchors)

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

        dst = self._rescale(src).transpose((2, 0, 1))[None, ...]

        if dst.shape != self.model._data_shapes[0].shape:
            self.exec_group.reshape([mx.io.DataDesc('data', dst.shape)], None)

        self.exec_group.data_arrays[0][0][1][:] = dst.astype(float32)
        self.exec_group.execs[0].forward(is_train=False)

        # print(f'inferance: {time.perf_counter() - timea}')

        return self._retina_solve()

    def detect(self, image, mode='nms'):
        out = self._retina_forward(image)
        detach = self._retina_detach(out)
        return getattr(self, f'_{mode}_wrapper')(detach)

    def workflow_inference(self, instream, shape):
        for source in instream:
            # st = time.perf_counter()

            frame = frombuffer(source, dtype=uint8).reshape(shape)

            out = self._retina_forward(frame)

            try:
                self.write_queue((frame, out))
            except Full:
                waitall()
                print('Frame queue full', file=sys.stderr)

            # print(f'workflow_inference: {time.perf_counter() - st}')

    def workflow_postprocess(self, outstream=None):
        for frame, out in self.read_queue:
            # st = time.perf_counter()
            detach = self._retina_detach(out)
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
                outstream(detach)


if __name__ == '__main__':
    import sys
    from numpy import prod

    FRAME_SHAPE = 480, 640, 3
    BUFFER_SIZE = prod(FRAME_SHAPE)

    read = sys.stdin.buffer.read
    write = sys.stdout.buffer.write
    camera = iter(partial(read, BUFFER_SIZE), b'')

    fd = MxnetDetectionModel("weights/16and32", 0, scale=.4, gpu=-1, margin=0.15)

    poster = Thread(target=fd.workflow_postprocess)
    poster.start()

    infer = Thread(target=fd.workflow_inference, args=(camera, FRAME_SHAPE,))
    infer.daemon = True
    infer.start()
