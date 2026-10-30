"""Text encoding of ARC tasks for a language model, and invertible task augmentations.

A grid is written one row per line, one digit per cell (Qwen tokenizes each digit as its own
token, so a 30x30 grid costs ~930 tokens). A prompt lists the demonstration pairs and then the
query input; the model completes the query output, terminated by END.
"""
from dataclasses import dataclass

import numpy as np

END = "END"


def grid_to_text(g):
    return "\n".join("".join(str(int(v)) for v in row) for row in g)


def text_to_grid(text):
    """Parse a model completion back into a grid; None if it is not a valid rectangle of digits."""
    rows = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line == END or not line:
            break
        if not line.isdigit():
            return None
        rows.append([int(c) for c in line])
    if not rows or len({len(r) for r in rows}) != 1 or len(rows) > 30 or len(rows[0]) > 30:
        return None
    return np.array(rows, dtype=np.int8)


def make_prompt(pairs, query):
    """Demonstration pairs plus the query input, ending where the model should write the output."""
    parts = ["Solve the puzzle: find the rule that turns each input grid into its output grid.\n"]
    for i, (x, y) in enumerate(pairs, 1):
        parts.append(f"Example {i} input:\n{grid_to_text(x)}\nExample {i} output:\n{grid_to_text(y)}\n{END}\n")
    parts.append(f"Test input:\n{grid_to_text(query)}\nTest output:\n")
    return "".join(parts)


def make_target(y):
    return f"{grid_to_text(y)}\n{END}"


# --- Augmentations ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Aug:
    """One of the 8 rotations/reflections, a colour permutation, and an example order."""
    dihedral: int            # 0..7: k = dihedral % 4 quarter turns, then transpose if dihedral >= 4
    colors: tuple            # colors[c] = new colour for colour c (0 stays 0: background)
    order: tuple | None = None

    def grid(self, g):
        g = np.rot90(g, self.dihedral % 4)
        if self.dihedral >= 4:
            g = g.T
        return np.asarray(self.colors, dtype=np.int8)[g]

    def invert(self, g):
        inverse = np.argsort(self.colors).astype(np.int8)
        g = inverse[g]
        if self.dihedral >= 4:
            g = g.T
        return np.ascontiguousarray(np.rot90(g, -(self.dihedral % 4)))

    def pairs(self, pairs):
        order = self.order if self.order is not None else range(len(pairs))
        return [(self.grid(pairs[i][0]), self.grid(pairs[i][1])) for i in order]


IDENTITY = Aug(0, tuple(range(10)))


def random_aug(rng, n_pairs, keep_background=True):
    perm = rng.permutation(np.arange(1, 10)) if keep_background else None
    colors = (0, *perm.tolist()) if keep_background else tuple(rng.permutation(10).tolist())
    return Aug(int(rng.integers(8)), colors, tuple(rng.permutation(n_pairs).tolist()))


def n_tokens_estimate(pairs, query):
    """Cells plus newlines plus fixed text; good enough to filter over-long prompts."""
    cells = sum(x.size + y.size + x.shape[0] + y.shape[0] for x, y in pairs) + query.size + query.shape[0]
    return cells + 40 * (len(pairs) + 1) + 30
