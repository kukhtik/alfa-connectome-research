"""WNN33 feature engineering — shared by the trainer and submission solution.

CAUTION, VERIFIED ON DATA (not from docs): the docs claim bid = p0..p10 and
ask = p11..p21, but a per-pair ordering scan on 500k real rows shows the
tightest consistently-ordered (i.e. true best-quote) pairs are:
    i0: bid col 2 (p2)  < ask col 13 (p13)   frac(diff>0)=1.0000, median 0.0850
    i1: bid col 54 (p2) < ask col 65 (p13)   frac(diff>0)=0.9992, median 0.1133
and NOT the documented max(bid-block)/min(ask-block) assignment (which fails
the bid<ask invariant entirely). The index is NOT a depth level, so block
aggregates like max/min over the block are meaningless — use the verified
quote columns instead. See tmp/find_quotes2.py for the self-checked probe.

IMPORTANT: numpy-only (no torch) so the identical code runs in solution.py.

All functions take arrays shaped (..., 112) and return (..., K) float32.
"""

import numpy as np

N_RAW = 112
EPS = 1e-6

# verified best-quote columns
I0_BEST_BID = 2
I0_BEST_ASK = 13
I1_BEST_BID = 54      # 52 + 2
I1_BEST_ASK = 65      # 52 + 13

# documented group ranges (still useful for volume/trade aggregates)
I0_P = slice(0, 22)
I0_V = slice(22, 44)
I0_DP = slice(44, 48)
I0_DV = slice(48, 52)
I1_P = slice(52, 74)
I1_V = slice(74, 96)
I1_DP = slice(96, 100)
I1_DV = slice(100, 104)
A_ALL = slice(104, 112)

FEATURE_NAMES = [
    "i0_best_bid", "i0_best_ask", "i0_spread", "i0_mid", "i0_rel_spread",
    "i1_best_bid", "i1_best_ask", "i1_spread", "i1_mid", "i1_rel_spread",
    "x_mid_diff", "x_spread_diff",
    "i0_v_sum", "i0_v_std", "i0_v_pos_sum", "i0_v_neg_sum", "i0_v_imb",
    "i1_v_sum", "i1_v_std", "i1_v_pos_sum", "i1_v_neg_sum", "i1_v_imb",
    "x_v_imb_diff",
    "i0_dp_mean", "i0_dp_std", "i0_dv_sum", "i0_dv_max", "i0_dv_pos_frac",
    "i1_dp_mean", "i1_dp_std", "i1_dv_sum", "i1_dv_max", "i1_dv_pos_frac",
    "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
]
N_DERIVED = len(FEATURE_NAMES)


def make_features(x):
    """Raw (..., 112) -> engineered (..., N_DERIVED) float32 (numpy only).

    Performance: this runs once per row at inference, so it is written to be
    fast on a single (1,1,112) input: preallocated output + in-place writes
    instead of np.stack over 41 arrays. See tmp/bench_feats.py — the naive
    np.stack version cost ~350 us/row (221 min for the test set, way over the
    60-min limit); the write-based version is ~10x cheaper.
    """
    x = np.asarray(x, dtype=np.float32)
    shp = x.shape[:-1]
    out = np.empty(shp + (N_DERIVED,), dtype=np.float32)
    f = out.reshape(-1, N_DERIVED)          # (P, N_DERIVED) view
    xf = x.reshape(-1, N_RAW)
    return _fill(xf, f).reshape(shp + (N_DERIVED,))


def _fill(xf, f):
    """Write the engineered columns into the preallocated view f. In-place."""
    EPSL = np.float32(EPS)
    b0b, b0a = xf[:, I0_BEST_BID], xf[:, I0_BEST_ASK]
    b1b, b1a = xf[:, I1_BEST_BID], xf[:, I1_BEST_ASK]
    s0 = b0a - b0b
    s1 = b1a - b1b
    m0 = 0.5 * (b0a + b0b)
    m1 = 0.5 * (b1a + b1b)
    v0 = xf[:, I0_V]
    v1 = xf[:, I1_V]
    v0sum = v0.sum(axis=1)
    v1sum = v1.sum(axis=1)
    v0p = np.clip(v0, 0, None).sum(axis=1)
    v0n = np.clip(v0, None, 0).sum(axis=1)
    v1p = np.clip(v1, 0, None).sum(axis=1)
    v1n = np.clip(v1, None, 0).sum(axis=1)
    dp0 = xf[:, I0_DP]
    dv0 = xf[:, I0_DV]
    dp1 = xf[:, I1_DP]
    dv1 = xf[:, I1_DV]

    cols = [
        b0b, b0a, s0, m0, s0 / (np.abs(m0) + EPSL),
        b1b, b1a, s1, m1, s1 / (np.abs(m1) + EPSL),
        m0 - m1, s0 - s1,
        v0sum, v0.std(axis=1), v0p, v0n, (v0p + v0n) / (v0p - v0n + EPSL),
        v1sum, v1.std(axis=1), v1p, v1n, (v1p + v1n) / (v1p - v1n + EPSL),
        (v0p + v0n) / (v0p - v0n + EPSL) - (v1p + v1n) / (v1p - v1n + EPSL),
        dp0.mean(axis=1), dp0.std(axis=1), dv0.sum(axis=1), dv0.max(axis=1),
        (dv0 > 0).mean(axis=1),
        dp1.mean(axis=1), dp1.std(axis=1), dv1.sum(axis=1), dv1.max(axis=1),
        (dv1 > 0).mean(axis=1),
    ]
    for i, c in enumerate(cols):
        f[:, i] = c
    f[:, 33:41] = xf[:, 104:112]
    return f
