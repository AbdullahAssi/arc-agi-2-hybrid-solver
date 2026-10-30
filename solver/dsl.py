"""Grid -> grid primitives.

Every primitive takes one grid and returns a new grid, or None when it does not apply
(the search prunes that branch). Primitives that invent a colour paint with MARK (10);
the search's colour-map fitting step later maps MARK to the colour the examples demand.
"""
import numpy as np
from scipy import ndimage

from .grid import background, bbox, objects

MARK = 10
MAX_SIDE = 30


def _fits(g):
    return g if g.size and max(g.shape) <= MAX_SIDE else None


# --- Geometry -------------------------------------------------------------------
def rot90(g): return np.rot90(g, 1)
def rot180(g): return np.rot90(g, 2)
def rot270(g): return np.rot90(g, 3)
def flip_h(g): return g[:, ::-1]
def flip_v(g): return g[::-1, :]
def transpose(g): return g.T
def antitranspose(g): return np.rot90(g, 2).T


# --- Cropping and selection ----------------------------------------------------
def crop_content(g):
    b = bbox(g != background(g))
    return None if b is None else g[b[0]:b[1], b[2]:b[3]]


def _crop_object(g, pick, by_color=True):
    objs = objects(g, by_color=by_color)
    if len(objs) < 2:
        return None
    mask, _ = pick(objs)
    r0, r1, c0, c1 = bbox(mask)
    return g[r0:r1, c0:c1]


def largest_object(g): return _crop_object(g, lambda o: max(o, key=lambda m: m[0].sum()))
def smallest_object(g): return _crop_object(g, lambda o: min(o, key=lambda m: m[0].sum()))
def largest_multicolor_object(g):
    return _crop_object(g, lambda o: max(o, key=lambda m: m[0].sum()), by_color=False)


def unique_color_object(g):
    """The object whose colour occurs in exactly one object (the odd one out)."""
    objs = objects(g)
    colors = [c for _, c in objs]
    odd = [(m, c) for m, c in objs if colors.count(c) == 1]
    if len(odd) != 1 or len(objs) < 3:
        return None
    r0, r1, c0, c1 = bbox(odd[0][0])
    return g[r0:r1, c0:c1]


def _half(g, axis, first):
    n = g.shape[axis]
    if n < 2:
        return None
    h = n // 2
    # Odd sizes: drop the middle row/column (often a separator line)
    sl = slice(0, h) if first else slice(n - h, n)
    return g[sl, :] if axis == 0 else g[:, sl]


def top_half(g): return _half(g, 0, True)
def bottom_half(g): return _half(g, 0, False)
def left_half(g): return _half(g, 1, True)
def right_half(g): return _half(g, 1, False)


# --- Scaling, tiling, mirroring --------------------------------------------------
def upscale2(g): return _fits(np.kron(g, np.ones((2, 2), dtype=g.dtype)))
def upscale3(g): return _fits(np.kron(g, np.ones((3, 3), dtype=g.dtype)))


def _downscale(g, k):
    h, w = g.shape
    if h % k or w % k or h == k and w == k:
        return None
    small = g[::k, ::k]
    return small if np.array_equal(np.kron(small, np.ones((k, k), dtype=g.dtype)), g) else None


def downscale2(g): return _downscale(g, 2)
def downscale3(g): return _downscale(g, 3)
def tile2x2(g): return _fits(np.tile(g, (2, 2)))
def tile3x3(g): return _fits(np.tile(g, (3, 3)))
def mirror_h(g): return _fits(np.hstack([g, g[:, ::-1]]))
def mirror_v(g): return _fits(np.vstack([g, g[::-1, :]]))
def mirror_4(g):
    top = np.hstack([g, g[:, ::-1]])
    return _fits(np.vstack([top, top[::-1, :]]))


def self_kron(g):
    """Place a copy of g at every non-background cell of g (fractal tiling)."""
    bg = background(g)
    out = np.kron((g != bg).astype(g.dtype), g)
    out[np.kron(g == bg, np.ones_like(g, dtype=bool))] = bg
    return _fits(out)


# --- Symmetry completion -----------------------------------------------------------
def _complete(g, transform):
    t = transform(g)
    if t.shape != g.shape:
        return None
    bg = background(g)
    out = g.copy()
    holes = (g == bg) & (t != bg)
    if not holes.any():
        return None
    out[holes] = t[holes]
    return out


def symmetrize_h(g): return _complete(g, flip_h)
def symmetrize_v(g): return _complete(g, flip_v)
def symmetrize_rot(g): return _complete(g, rot180)
def symmetrize_diag(g): return _complete(g, transpose) if g.shape[0] == g.shape[1] else None


# --- Gravity ----------------------------------------------------------------------------
def _gravity_down(g):
    # Stable sort on "is coloured" moves background cells up, keeping colour order
    order = np.argsort(g != background(g), axis=0, kind="stable")
    return np.take_along_axis(g, order, axis=0)


def gravity_down(g): return _gravity_down(g)
def gravity_up(g): return _gravity_down(g[::-1])[::-1]
def gravity_right(g): return _gravity_down(g.T).T
def gravity_left(g): return _gravity_down(g[:, ::-1].T).T[:, ::-1]


# --- Object-level edits -----------------------------------------------------------------
def remove_noise(g):
    """Delete single-cell objects."""
    bg = background(g)
    out = g.copy()
    removed = False
    for mask, _ in objects(g, diagonal=False):
        if mask.sum() == 1:
            out[mask] = bg
            removed = True
    return out if removed else None


def keep_largest(g):
    objs = objects(g, by_color=False)
    if len(objs) < 2:
        return None
    mask, _ = max(objs, key=lambda m: m[0].sum())
    return np.where(mask, g, background(g)).astype(g.dtype)


def fill_holes(g):
    """Paint background regions fully enclosed by non-background cells with MARK."""
    bg = background(g)
    holes = ndimage.binary_fill_holes(g != bg) & (g == bg)
    if not holes.any():
        return None
    out = g.copy()
    out[holes] = MARK
    return out


def outline(g):
    """Mark the interior of solid objects, leaving their 1-cell border."""
    bg = background(g)
    inner = ndimage.binary_erosion(g != bg)
    if not inner.any():
        return None
    out = g.copy()
    out[inner] = MARK
    return out


def color_by_size_rank(g):
    """Recolour each object by its size rank (1 = largest), so colour maps can pick ranks."""
    objs = objects(g, by_color=False)
    if not 2 <= len(objs) <= 9:
        return None
    out = g.copy()
    for rank, (mask, _) in enumerate(sorted(objs, key=lambda m: -m[0].sum()), 1):
        out[mask] = rank
    return out


def _rays(g, horizontal):
    """Fill the background of each row (column) with the colour of its first coloured cell."""
    lines = g if horizontal else g.T
    bg = background(g)
    fg = lines != bg
    has = fg.any(axis=1)
    if not has.any():
        return None
    fill = lines[np.arange(len(lines)), np.argmax(fg, axis=1)]
    out = np.where(~fg & has[:, None], fill[:, None], lines).astype(g.dtype)
    out = out if horizontal else out.T
    return out if not np.array_equal(out, g) else None


def rays_h(g): return _rays(g, True)
def rays_v(g): return _rays(g, False)


# --- Splitting a grid in two and combining the halves -------------------------------------
def _split_parts(g):
    """Split on a full-height/width separator line, else into equal halves."""
    h, w = g.shape
    bg = background(g)
    # A separator splits the grid into equal halves, so it can only sit in the exact middle
    if w % 2 and w >= 3 and (g[:, w // 2] == g[0, w // 2]).all() and g[0, w // 2] != bg:
        return g[:, : w // 2], g[:, w // 2 + 1:]
    if h % 2 and h >= 3 and (g[h // 2] == g[h // 2, 0]).all() and g[h // 2, 0] != bg:
        return g[: h // 2], g[h // 2 + 1:]
    if w % 2 == 0 and w >= 2:
        return g[:, : w // 2], g[:, w // 2:]
    if h % 2 == 0 and h >= 2:
        return g[: h // 2], g[h // 2:]
    return None


def _logic(g, op):
    parts = _split_parts(g)
    if parts is None:
        return None
    a, b = parts
    bg = background(g)
    mask = op(a != bg, b != bg)
    return np.where(mask, MARK, bg).astype(g.dtype)


def logic_and(g): return _logic(g, np.logical_and)
def logic_or(g): return _logic(g, np.logical_or)
def logic_xor(g): return _logic(g, np.logical_xor)
def logic_nor(g): return _logic(g, lambda a, b: ~(a | b))


def overlay_halves(g):
    """Stack the two halves, the second half's non-background cells on top."""
    parts = _split_parts(g)
    if parts is None:
        return None
    a, b = parts
    bg = background(g)
    return np.where(b != bg, b, a)


# --- Drawing new cells -------------------------------------------------------------------------
def _halo(g, diagonal):
    bg = background(g)
    grown = ndimage.binary_dilation(g != bg, structure=np.ones((3, 3)) if diagonal else None)
    ring = grown & (g == bg)
    if not ring.any():
        return None
    out = g.copy()
    out[ring] = MARK
    return out


def halo8(g): return _halo(g, True)
def halo4(g): return _halo(g, False)


def fill_bbox(g):
    """Mark background cells inside each object's bounding box."""
    bg = background(g)
    out = g.copy()
    for mask, _ in objects(g, by_color=False):
        r0, r1, c0, c1 = bbox(mask)
        box = out[r0:r1, c0:c1]
        box[box == bg] = MARK
    return out if not np.array_equal(out, g) else None


def connect_pairs(g):
    """Join same-coloured cells sharing a row or column with a line of that colour."""
    bg = background(g)
    out = g.copy()
    for c in np.unique(g):
        if c == bg:
            continue
        for lines in (out, out.T):
            src = g if lines is out else g.T
            for i in range(src.shape[0]):
                idx = np.where(src[i] == c)[0]
                if len(idx) >= 2:
                    seg = lines[i, idx[0]:idx[-1] + 1]
                    seg[seg == bg] = c
    return out if not np.array_equal(out, g) else None


def complete_pattern(g):
    """Fill cells of one 'unknown' colour by the smallest consistent 2-D period."""
    h, w = g.shape
    if h * w < 16:
        return None
    rr, cc = np.indices(g.shape)
    periods = sorted(((a, b) for a in range(1, min(h, 8) + 1) for b in range(1, min(w, 8) + 1)
                      if a < h or b < w), key=lambda p: p[0] * p[1])
    counts = np.bincount(g.ravel(), minlength=11)
    # The unknown colour is a minority colour (at most 40% of cells); try the two rarest
    candidates = [c for c in np.argsort(counts) if 0 < counts[c] <= 0.4 * g.size][:2]
    for unknown in candidates:
        known = g != unknown
        r, c, v = rr[known], cc[known], g[known].astype(np.int64)
        for ph, pw in periods:
            seen = np.zeros((ph * pw, 11), bool)
            seen[(r % ph) * pw + c % pw, v] = True
            if (seen.sum(1) == 1).all():  # every slot seen, and seen with a single colour
                tile = seen.argmax(1).reshape(ph, pw)
                return np.tile(tile, (h // ph + 1, w // pw + 1))[:h, :w].astype(g.dtype)
    return None


# --- Summaries of the whole grid -----------------------------------------------------------------
def frame_interior(g):
    """Crop the inside of the largest single-colour rectangular frame."""
    best = None
    for mask, _ in objects(g, diagonal=False):
        r0, r1, c0, c1 = bbox(mask)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        border = np.zeros_like(mask)
        border[r0:r1, c0:c1] = True
        border[r0 + 1:r1 - 1, c0 + 1:c1 - 1] = False
        if np.array_equal(mask & border, border) and (best is None or mask.sum() > best[0]):
            best = (mask.sum(), g[r0 + 1:r1 - 1, c0 + 1:c1 - 1])
    return None if best is None else best[1]


def _cells(g):
    """Split a grid divided by full separator lines into its cells (list of rows of grids)."""
    h, w = g.shape
    for c in np.unique(g):
        rows = [i for i in range(h) if (g[i] == c).all()]
        cols = [j for j in range(w) if (g[:, j] == c).all()]
        if not rows and not cols:
            continue
        rb = [-1] + rows + [h]
        cb = [-1] + cols + [w]
        cells = [[g[a + 1:b, p + 1:q] for p, q in zip(cb, cb[1:]) if q - p > 1]
                 for a, b in zip(rb, rb[1:]) if b - a > 1]
        if cells and cells[0] and len(cells) * len(cells[0]) >= 2 \
                and len({x.shape for row in cells for x in row}) == 1:
            return cells, int(c)
    return None


def cells_summary(g):
    """One pixel per cell: the cell's most common non-background colour."""
    split = _cells(g)
    if split is None:
        return None
    cells, _ = split
    bg = background(g)

    def dominant(cell):
        counts = np.bincount(cell.ravel(), minlength=11)
        counts[bg] = 0
        return counts.argmax() if counts.any() else bg

    return np.array([[dominant(x) for x in row] for row in cells], dtype=g.dtype)


def odd_cell(g):
    """The one cell whose content differs from all the others."""
    split = _cells(g)
    if split is None:
        return None
    flat = [x for row in split[0] for x in row]
    keys = [x.tobytes() for x in flat]
    odd = [x for x, k in zip(flat, keys) if keys.count(k) == 1]
    return odd[0] if len(odd) == 1 and len(flat) >= 3 else None


def dominant_color(g):
    counts = np.bincount(g.ravel(), minlength=11)
    counts[background(g)] = 0
    return np.array([[counts.argmax()]], dtype=g.dtype) if counts.any() else None


PRIMITIVES = {f.__name__: f for f in [
    rot90, rot180, rot270, flip_h, flip_v, transpose, antitranspose,
    crop_content, largest_object, smallest_object, largest_multicolor_object, unique_color_object,
    top_half, bottom_half, left_half, right_half,
    upscale2, upscale3, downscale2, downscale3, tile2x2, tile3x3, mirror_h, mirror_v, mirror_4, self_kron,
    symmetrize_h, symmetrize_v, symmetrize_rot, symmetrize_diag,
    gravity_down, gravity_up, gravity_left, gravity_right,
    remove_noise, keep_largest, fill_holes, outline, color_by_size_rank, rays_h, rays_v,
    logic_and, logic_or, logic_xor, logic_nor, overlay_halves,
    halo8, halo4, fill_bbox, connect_pairs, complete_pattern,
    frame_interior, cells_summary, odd_cell, dominant_color,
]}
