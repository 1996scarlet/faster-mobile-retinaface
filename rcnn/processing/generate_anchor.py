"""
Generate base anchors on index 0
"""
import sys
import numpy as np
from ..cython.anchors import anchors_cython
from mxnet.ndarray import exp

anchors = {
    32: {'SCALES': (32, 16), 'BASE_SIZE': 16, 'RATIOS': (1., ), 'ALLOWED_BORDER': 0},
    16: {'SCALES': (8, 4), 'BASE_SIZE': 16, 'RATIOS': (1., ), 'ALLOWED_BORDER': 0},
    # '8': {'SCALES': (2, 1), 'BASE_SIZE': 16, 'RATIOS': (1., ), 'ALLOWED_BORDER': 9999},
}


def anchors_plane(feat_h, feat_w, stride, base_anchor):
    return anchors_cython(feat_h, feat_w, stride, base_anchor)


def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6), stride=16, dense_anchor=False):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    if dense_anchor:
        assert stride % 2 == 0
        anchors2 = anchors.copy()
        anchors2[:, :] += int(stride/2)
        anchors = np.vstack((anchors, anchors2))
    # print('GA',base_anchor.shape, ratio_anchors.shape, anchors.shape)
    return anchors


def generate_anchors_fpn(dense_anchor=False, cfg=anchors):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    # cfg = {k: cfg[k] for k in sorted(cfg.keys(), reverse=True)}

    anchors = [generate_anchors(value['BASE_SIZE'],
                                np.array(value['RATIOS']),
                                np.array(value['SCALES']),
                                stride,
                                dense_anchor) for stride, value in cfg.items()]

    return dict(zip(cfg.keys(), np.asarray(anchors, dtype=np.float32)))


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def nonlinear_pred(boxes, box_deltas):
    if boxes.size:
        ctr_x, ctr_y, widths, heights = boxes.T
        widths -= ctr_x
        heights -= ctr_y

        widths += 1.0
        heights += 1.0

        dx, dy, dw, dh = box_deltas.T

        dx *= widths
        dx += ctr_x
        dx += 0.5 * widths

        dy *= heights
        dy += ctr_y
        dy += 0.5 * heights

        np.exp(dh, out=dh)
        dh *= heights
        dh -= 1.0
        dh *= 0.5

        np.exp(dw, out=dw)
        dw *= widths
        dw -= 1.0
        dw *= 0.5

        dx -= dw
        dw *= 2.0
        dw += dx

        dy -= dh
        dh *= 2.0
        dh += dy

# def nonlinear_pred(boxes, box_deltas):
#     if boxes.size:
#         ctr_x, ctr_y, widths, heights = boxes.T
#         widths -= ctr_x
#         heights -= ctr_y

#         widths += 1.0
#         heights += 1.0

#         dx, dy, dw, dh = box_deltas.T

#         dx *= widths
#         dx += ctr_x
#         dx += 0.5 * widths

#         dy *= heights
#         dy += ctr_y
#         dy += 0.5 * heights

#         exp(dh, out=dh)
#         dh *= heights
#         dh -= 1.0
#         dh *= 0.5

#         exp(dw, out=dw)
#         dw *= widths
#         dw -= 1.0
#         dw *= 0.5

#         dx -= dw
#         dw *= 2.0
#         dw += dx

#         dy -= dh
#         dh *= 2.0
#         dh += dy