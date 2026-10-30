"""Fixed-length description of a task's train pairs, used as input to the learned prior."""
import numpy as np

from .dsl import _split_parts
from .grid import background, objects


def _contains(big, small):
    """True if `small` occurs as a contiguous sub-grid of `big`."""
    H, W = big.shape
    h, w = small.shape
    if h > H or w > W:
        return False
    return any(np.array_equal(big[r:r + h, c:c + w], small)
               for r in range(H - h + 1) for c in range(W - w + 1))


def _symmetries(g):
    return [np.array_equal(g, g[:, ::-1]), np.array_equal(g, g[::-1]),
            np.array_equal(g, np.rot90(g, 2)), g.shape[0] == g.shape[1] and np.array_equal(g, g.T)]


def pair_features(x, y):
    hx, wx = x.shape
    hy, wy = y.shape
    bx, by = background(x), background(y)
    cx, cy = set(np.unique(x)), set(np.unique(y))
    same = x.shape == y.shape
    changed = (x != y) if same else None
    ox, oy = objects(x, by_color=False), objects(y, by_color=False)
    f = [
        hy / hx, wy / wx, float(same), float(hy * wy < hx * wx), float(hy * wy > hx * wx),
        float(hy == 2 * hx), float(wy == 2 * wx), float(hy == 3 * hx), float(wy == 3 * wx),
        float(2 * hy in (hx, hx - 1)), float(2 * wy in (wx, wx - 1)), float(hy == hx * hx and wy == wx * wx),
        float((hx, wx) == (wy, hy) and not same),
        len(cx), len(cy), len(cy - cx), len(cx - cy), float(bx == by),
        (x != bx).mean(), (y != by).mean(),
        changed.mean() if same else -1.0,
        (changed & (x == bx)).sum() / max(changed.sum(), 1) if same else -1.0,
        (changed & (y == by)).sum() / max(changed.sum(), 1) if same else -1.0,
        len(ox), len(oy), float(len(oy) < len(ox)),
        float(_contains(x, y)), float(_contains(y, x)),
        float(_split_parts(x) is not None and _split_parts(x)[0].shape == y.shape),
    ]
    f += [float(s) for s in _symmetries(x)] + [float(s) for s in _symmetries(y)]
    if same:
        f += [float(np.array_equal(y, t)) for t in (np.rot90(x), np.rot90(x, 2), x[:, ::-1], x[::-1])]
    else:
        f += [0.0] * 4
    return f


N_FEATURES = len(pair_features(np.zeros((2, 2), np.int8), np.ones((2, 2), np.int8)))


def task_features(pairs):
    """Mean and max of pair features over a task's (input, output) pairs."""
    m = np.array([pair_features(x, y) for x, y in pairs], dtype=np.float32)
    return np.concatenate([m.mean(0), m.max(0), m.min(0)])
