"""Best-first program search over compositions of DSL primitives.

A node is a program (sequence of primitive names) together with the grids it produces on
every train and test input. Nodes are expanded cheapest-first, where a program's cost is
the sum of its primitives' costs: uniform costs give breadth-first search, costs from the
learned prior give guided search. Nodes producing grids already seen are dropped.
"""
import heapq
import itertools
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .dsl import PRIMITIVES
from .grid import background, bbox, key, objects


@dataclass
class Result:
    attempts: list = field(default_factory=list)   # per test input: up to 2 predicted grids
    programs: list = field(default_factory=list)   # programs behind attempt 1, attempt 2
    nodes: int = 0                                  # nodes expanded in total
    first_hit: int | None = None                    # nodes expanded when first solution found


def fit_color_map(preds, targets):
    """Return a colour mapping that turns every pred into its target, or None."""
    if any(p.shape != y.shape for p, y in zip(preds, targets)):
        return None
    # Every (src, dst) colour pair that occurs; a valid map has one dst per src
    pairs = np.unique(np.concatenate([p.ravel().astype(np.int16) * 11 + y.ravel() for p, y in zip(preds, targets)]))
    src, dst = pairs // 11, pairs % 11
    if len(np.unique(src)) != len(src):
        return None
    return dict(zip(src.tolist(), dst.tolist()))


def apply_color_map(g, mapping):
    out = g.copy()
    for src in np.unique(g):
        dst = mapping.get(int(src), int(src))
        if dst > 9:
            return None
        out[g == src] = dst
    return out


def solve(task, costs=None, max_depth=3, max_nodes=20_000, time_limit=30.0):
    """Search for up to two distinct programs that fit all train pairs of `task`."""
    costs = costs or {name: 1.0 for name in PRIMITIVES}
    train_in = [x for x, _ in task["train"]]
    train_out = [y for _, y in task["train"]]
    test_in = [x for x, _ in task["test"]]
    n_train = len(train_in)

    result = Result()
    seen_states, seen_preds = set(), set()
    tie = itertools.count()
    frontier = [(0.0, next(tie), (), tuple(train_in + test_in))]
    start = time.perf_counter()

    while frontier and result.nodes < max_nodes and time.perf_counter() - start < time_limit:
        cost, _, program, grids = heapq.heappop(frontier)
        result.nodes += 1

        preds = _check(grids[:n_train], grids[n_train:], train_out)
        if preds is not None:
            pred_key = tuple(key(p) for p in preds)
            if pred_key not in seen_preds:
                seen_preds.add(pred_key)
                result.programs.append(program)
                if result.first_hit is None:
                    result.first_hit = result.nodes
                result.attempts = [a + [p] for a, p in zip(result.attempts or [[] for _ in preds], preds)]
                if len(result.programs) == 2:
                    break

        if len(program) == max_depth:
            continue
        for name, fn in PRIMITIVES.items():
            new = []
            for g in grids:
                out = fn(g)
                if out is None:
                    break
                new.append(np.ascontiguousarray(out))
            else:
                state = tuple(key(g) for g in new)
                if state not in seen_states:
                    seen_states.add(state)
                    heapq.heappush(frontier, (cost + costs[name], next(tie), program + (name,), tuple(new)))

    if not result.attempts:  # nothing fits: fall back to echoing the input
        result.attempts = [[x] for x in test_in]
    return result


def _object_props(g):
    """Objects of g with candidate properties that might decide their output colour."""
    objs = objects(g)
    sizes = sorted({int(m.sum()) for m, _ in objs}, reverse=True)
    shapes = []
    for m, _ in objs:
        r0, r1, c0, c1 = bbox(m)
        shapes.append(m[r0:r1, c0:c1].tobytes() + bytes([r1 - r0]))
    out = []
    for (m, color), shape in zip(objs, shapes):
        r0, r1, c0, c1 = bbox(m)
        size = int(m.sum())
        out.append((m, {
            "color": color,
            "size": size,
            "size_rank": sizes.index(size),
            "is_largest": size == sizes[0],
            "is_smallest": size == sizes[-1],
            "touches_border": bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()),
            "has_hole": bool((ndimage.binary_fill_holes(m) & ~m).any()),
            "shape": shape,
            "shape_count": shapes.count(shape),
            "height": r1 - r0,
            "width": c1 - c0,
            "is_rect": size == (r1 - r0) * (c1 - c0),
        }))
    return out


def fit_object_recolor(preds, targets):
    """Find an object property whose value determines each object's output colour.

    Requires outputs to differ from preds only by recolouring whole objects. Returns
    (property, {value: colour}) or None.
    """
    samples = []
    for p, y in zip(preds, targets):
        if p.shape != y.shape:
            return None
        bg = background(p)
        if not np.array_equal(p == bg, y == bg):
            return None
        covered = np.zeros(p.shape, bool)
        for m, props in _object_props(p):
            dst = np.unique(y[m])
            if len(dst) != 1:
                return None
            covered |= m
            samples.append((props, int(dst[0])))
        if not np.array_equal(p[~covered], y[~covered]):
            return None
    if not samples or all(props["color"] == dst for props, dst in samples):
        return None
    for prop in samples[0][0]:
        rule = {}
        if all(rule.setdefault(props[prop], dst) == dst for props, dst in samples):
            if len(set(rule.values())) > 1 or prop == "color":
                return prop, rule
    return None


def apply_object_recolor(g, fit):
    prop, rule = fit
    out = g.copy()
    for m, props in _object_props(g):
        if props[prop] not in rule:
            return None
        out[m] = rule[props[prop]]
    return out


def _check(train_preds, test_preds, train_out):
    """Return one prediction per test input if the program fits all train pairs, else None."""
    if all(p.shape == y.shape and np.array_equal(p, y) for p, y in zip(train_preds, train_out)):
        outs = list(test_preds)
    elif (mapping := fit_color_map(train_preds, train_out)) is not None:
        outs = [apply_color_map(p, mapping) for p in test_preds]
    elif (fit := fit_object_recolor(train_preds, train_out)) is not None:
        outs = [apply_object_recolor(p, fit) for p in test_preds]
    else:
        return None
    if any(o is None or o.max() > 9 for o in outs):
        return None
    return outs
