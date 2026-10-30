"""Task loading and low-level grid helpers. Grids are 2-D numpy int8 arrays."""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

DATA = Path(__file__).resolve().parents[1] / "data" / "arc-agi-2" / "data"


def load_split(split):
    """Return {task_id: {"train": [(in, out), ...], "test": [(in, out|None), ...]}}."""
    tasks = {}
    for path in sorted((DATA / split).glob("*.json")):
        raw = json.loads(path.read_text())
        tasks[path.stem] = {
            part: [(np.array(p["input"], dtype=np.int8),
                    np.array(p["output"], dtype=np.int8) if "output" in p else None)
                   for p in raw[part]]
            for part in ("train", "test")
        }
    return tasks


# Per-process memo: many primitives analyse the same grid at every search node.
# Cached masks are shared, so callers must treat them as read-only.
_CACHE = {}
_CACHE_LIMIT = 20_000


def _memo(tag, g, compute):
    k = (tag, g.shape, g.tobytes())
    if k not in _CACHE:
        if len(_CACHE) > _CACHE_LIMIT:
            _CACHE.clear()
        _CACHE[k] = compute()
    return _CACHE[k]


def background(g):
    """Most common colour, treated as background."""
    return _memo("bg", g, lambda: int(np.bincount(g.ravel(), minlength=10).argmax()))


def bbox(mask):
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any():
        return None
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return r0, r1 + 1, c0, c1 + 1


def objects(g, diagonal=True, by_color=True):
    """Connected components of non-background cells as (mask, colour-or-None) pairs."""
    return _memo(("obj", diagonal, by_color), g, lambda: _objects(g, diagonal, by_color))


def _objects(g, diagonal, by_color):
    bg = background(g)
    structure = np.ones((3, 3)) if diagonal else None
    found = []
    if by_color:
        for c in np.unique(g):
            if c == bg:
                continue
            labels, n = ndimage.label(g == c, structure=structure)
            found += [(labels == i, int(c)) for i in range(1, n + 1)]
    else:
        labels, n = ndimage.label(g != bg, structure=structure)
        found = [(labels == i, None) for i in range(1, n + 1)]
    return found


def key(g):
    """Hashable fingerprint of a grid, used to deduplicate search states."""
    return g.shape, g.tobytes()
