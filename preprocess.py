"""Turns a drawing (a grid of ink intensities 0..1) into the 8x8 input of the model.

The SAME algorithm is implemented in JavaScript in app_template.html,
so the model sees the same kind of input in training and in the browser.

Steps: find the bounding box of the ink -> fit it into the centre 6x6 area of an 8x8 grid
(keeping the aspect ratio) -> average the ink inside every cell -> scale so the darkest cell is 16
(the same range as the scikit-learn digits).
"""
import numpy as np


def _round(v):
    return int(np.floor(v + 0.5))   # same rounding as Math.floor(v + 0.5) in JavaScript


def preprocess(ink):
    ink = np.asarray(ink, dtype=float)
    m = ink.max()
    if m <= 0:
        return None
    ys, xs = np.where(ink > 0.1 * m)
    x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x_max - x_min + 1, y_max - y_min + 1
    side = max(w, h)
    s8 = side * 8 / 6
    cx, cy = (x_min + x_max + 1) / 2, (y_min + y_max + 1) / 2
    b = s8 / 8
    x0, y0 = cx - s8 / 2, cy - s8 / 2
    pad = int(np.ceil(s8)) + 3
    p = np.pad(ink, pad)
    out = np.zeros((8, 8))
    for i in range(8):
        ys0 = _round(y0 + i * b)
        ys1 = max(ys0 + 1, _round(y0 + (i + 1) * b))
        for j in range(8):
            xs0 = _round(x0 + j * b)
            xs1 = max(xs0 + 1, _round(x0 + (j + 1) * b))
            out[i, j] = p[ys0 + pad:ys1 + pad, xs0 + pad:xs1 + pad].mean()
    mx = out.max()
    if mx <= 0:
        return None
    return (16 * out / mx).ravel()
