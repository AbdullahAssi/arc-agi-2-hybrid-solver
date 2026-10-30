"""Task descriptors shared by the analysis notebooks."""
import numpy as np

from solver.grid import background


def category(task):
    """Coarse task type from how outputs relate to inputs."""
    pairs = task["train"]
    if all(x.shape == y.shape for x, y in pairs):
        added = np.mean([((x != y) & (x == background(x))).sum() / max((x != y).sum(), 1) for x, y in pairs])
        if added > 0.95:
            return "same size: adds cells"
        if added < 0.05:
            return "same size: recolour/remove"
        return "same size: mixed edits"
    if all(y.size < x.size for x, y in pairs):
        return "smaller output"
    if all(y.size > x.size for x, y in pairs):
        return "larger output"
    return "mixed sizes"
