"""224 inputs: the raw 112 features plus their first difference vs the previous row.

David Alami (0.6645, Alfa Connectome): "Standardise the 112 features ... and append
the change from the previous row (224 inputs)". The first row of a sequence has a
zero delta; state resets when seq_ix changes.

Training path (vectorised) and inference path (row-by-row) must agree exactly;
tests/test_delta224.py checks that.
"""
import numpy as np

N_RAW = 112
N_DELTA = 2 * N_RAW


def delta224_np(x: np.ndarray) -> np.ndarray:
    """(T, 112) raw sequence -> (T, 224) [raw, delta]."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != N_RAW:
        raise ValueError(f"expected (T,{N_RAW}), got {x.shape}")
    d = np.empty_like(x)
    d[0] = 0.0
    if x.shape[0] > 1:
        d[1:] = x[1:] - x[:-1]
    return np.concatenate([x, d], axis=1).astype(np.float32, copy=False)


class OnlineDelta224:
    """Row-by-row equivalent of delta224_np."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.prev = None

    def update(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.shape != (N_RAW,):
            raise ValueError(f"expected ({N_RAW},), got {x.shape}")
        d = np.zeros(N_RAW, dtype=np.float32) if self.prev is None else x - self.prev
        self.prev = x
        return np.concatenate([x, d]).astype(np.float32, copy=False)
