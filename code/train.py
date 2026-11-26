#!/usr/bin/env python3
"""WNN33 m1 trainer: stateful 2-layer GRU, full 20k-step sequences, GPU (WSL venv).

Trains on complete 20,000-step sequences (no BPTT truncation by default),
batches of sequences read from parquet row groups with a thread prefetcher,
and exports a stateful ONNX model matching the starterpack signature:

  features[1,1,112] + hidden_0[1,1,H] + hidden_1[1,1,H]
  -> prediction[1,1,2] + next_hidden_0[1,1,H] + next_hidden_1[1,1,H]

Run:
  wsl -d Ubuntu -- bash -lc '~/wnn_venv/bin/python /mnt/d/wnn_work/train.py \
      --data /mnt/d/wnn_work/data/train.parquet --out /mnt/d/wnn_work/models --epochs 2'
"""
import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from twin_arch import TwinEncGRU, TwinGRU  # noqa: E402
except Exception:  # pragma: no cover
    # train.py may be imported with src/ not on sys.path (e.g. tests)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from twin_arch import TwinEncGRU, TwinGRU  # noqa: E402

try:
    from slstm import SLSTM, StatefulSLSTM  # noqa: E402
except Exception:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from slstm import SLSTM, StatefulSLSTM  # noqa: E402

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/mnt/d/wnn_starterpack_partial")
sys.path.insert(0, "/mnt/d/wnn_work")
from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH, WARMUP, METRIC_CLIP  # noqa: E402
from feat_eng import (  # noqa: E402
    A_ALL, EPS, I0_BEST_ASK, I0_BEST_BID, I0_DP, I0_DV, I0_V,
    I1_BEST_ASK, I1_BEST_BID, I1_DP, I1_DV, I1_V, N_DERIVED as N_DERIVED_ENG,
)

N_FEATURES = len(FEATURE_COLUMNS)
TARGET_COLUMNS = ("t0", "t1")
# TZ-6 (26.10): u-basis. u+ = (t0+t1)/sqrt2 (mode that c3 fails), u- = (t0-t1)/sqrt2
# (mode whose variance DOMINATES the sum because WP(u-) is far higher). Exact
# orthogonal 2x2 rotation, so the (t0,t1) export is a linear map of the two heads
# and the ONNX contract (3-in/3-out) does not change.
INV_SQRT2 = float(2.0 ** -0.5)


class PersistFeat(nn.Module):
    """D2: append one raw feature column as an extra channel, inside the graph.

    The column index is chosen at construction (a3 = 107 by default when enabled).
    Because the transform lives inside the ONNX graph, the exported model keeps the
    baseline's 112-wide input contract: the platform still feeds 112 columns and the
    graph appends the persistence channel itself.

    The appended column is standardized with the SAME mean/std row as the main
    features (norm_mean/norm_std), so it is on the same scale as everything else.
    """

    def __init__(self, col):
        super().__init__()
        self.col = int(col)

    def forward(self, x):                     # x: (..., F)
        extra = x[..., self.col:self.col + 1]
        return torch.cat([x, extra], dim=-1)


class FeatMix(nn.Module):
    """F2 (07.10.2026, MA2): zero-init residual feature-mixing bottleneck.

    x' = x + W2 @ GELU(W1 @ x + b1) + b2          # W2, b2 zero-initialised
    At init exactly the identity -> training starts from the champion behaviour
    and moves away only if nonlinear cross-feature combinations help. The fix
    for the N6 gate's failure mode (sigmoid(0)=0.5 halved the state from step 0).
    112 -> mix_hid -> 112, ~7.3k params, ~3% per-row extra compute (measured).
    """

    def __init__(self, n_features=N_FEATURES, mix_hid=32):
        super().__init__()
        self.lin1 = nn.Linear(n_features, mix_hid)
        self.lin2 = nn.Linear(mix_hid, n_features)
        nn.init.zeros_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x):
        return x + self.lin2(torch.nn.functional.gelu(self.lin1(x)))


class FeatGate(nn.Module):
    """N6-cell: feature-axis mixing - gate the GRU output by a function of the raw row.

    WHY (N5, literature): TLOB uses attention over the FEATURE axis and CVML uses
    explicit cross-variate mixing; both report that inter-feature mixing is the axis
    LOB models are missing. Our GRU mixes features only once, inside weight_ih. A
    gate computed from the raw tick lets 'which features matter right now' modulate
    the hidden state every step:

        a = sigmoid(W_g @ x_t + b_g)     # (B,T,hid), from the RAW row
        h' = h * a                       # per-unit gate

    The gate reads the row AFTER normalization (same tensor the GRU sees), so the
    numbers it multiplies are on the standard scale. Params: 112*hid + hid.
    Measured cost on 1-thread CPU: x1.2-1.6 per row (FM probe) -> design the
    ensemble as a 2-member class, or use a subset gate.
    """

    def __init__(self, hid, n_features, keep=None):
        super().__init__()
        self.keep = list(keep) if keep is not None else None
        n_in = len(self.keep) if self.keep is not None else n_features
        self.lin = nn.Linear(n_in, hid)

    def forward(self, x):                     # x: (B,T,F) normalized
        src = x[..., self.keep] if self.keep is not None else x
        return torch.sigmoid(self.lin(src))   # (B,T,hid)



class StatefulGRU(nn.Module):
    """GRU + linear head; optional input standardization; single-step.

    With use_eng the raw 112 features are transformed inside the graph
    (FeatureTransform), so the exported ONNX takes the same 112-wide input as
    the baseline and runs the whole pipeline in C++.
    """

    def __init__(self, hid=96, layers=2, n_features=N_FEATURES, use_eng=False,
                 head_raw=False, head_mlp=False, persist_col=-1, feat_gate=False, feat_mix=False,
                 u_basis=False, gamma_gate=False, k_head=False, k_lambda=0.5, front_dim=0):
        super().__init__()
        self.hid = hid
        self.layers = layers
        self.use_eng = use_eng
        self.head_raw = head_raw          # D1: head sees [h, raw tick] not just [h]
        self.persist_col = persist_col    # D2: append this raw column as an extra input
        self.feat_gate = feat_gate        # N6: gate the GRU output by the raw row
        self.u_basis = u_basis            # TZ-6: heads are (u+,u-); export is their rotation
        self.gamma_gate = gamma_gate      # TZ-6: scalar gate g(h_t) in (0,1) on u+ only
        self.feat = FeatureTransform() if use_eng else None
        self.persist = PersistFeat(persist_col) if persist_col >= 0 else None
        self.n_features = N_DERIVED_ENG if use_eng else n_features
        if self.persist is not None:
            self.n_features += 1
        self.front_dim = int(front_dim)
        self.front = None
        self.post_ln = None
        self.head_d = None
        _gru_in = self.n_features
        if self.front_dim > 0:
            self.front = nn.Sequential(nn.Linear(self.n_features, self.front_dim),
                                       nn.GELU(), nn.LayerNorm(self.front_dim))
            _gru_in = self.front_dim
        self.gru = nn.GRU(_gru_in, hid, num_layers=layers, batch_first=True)
        self.gate = FeatGate(hid, self.n_features) if feat_gate else None
        self.head = (nn.Sequential(nn.Linear(hid + N_FEATURES, hid), nn.ReLU(),
                                   nn.Linear(hid, 2))
                     if head_mlp else
                     nn.Linear(hid + (N_FEATURES if head_raw else 0), 2))
        assert head_raw or not head_mlp, "head_mlp requires head_raw"
        self.head_mlp = head_mlp
        if self.front_dim > 0:
            self.post_ln = nn.LayerNorm(hid + self.front_dim)
            self.head_d = nn.Sequential(nn.Linear(hid + self.front_dim, self.front_dim),
                                        nn.GELU(), nn.Linear(self.front_dim, 2))
        # F2: created AFTER head (RNG-stream order -> champion init exact).
        self.feat_mix = FeatMix(self.n_features) if feat_mix else None
        # TZ-6 gate: created LAST and ZERO-initialised -> gamma == 0.5 exactly at
        # step 0 (neutral start, spec) and the champion RNG stream above is intact.
        self.gamma = nn.Linear(hid, 1) if gamma_gate else None
        if self.gamma is not None:
            nn.init.zeros_(self.gamma.weight)
            nn.init.zeros_(self.gamma.bias)
        # K-HEAD (26.10): the streaming graph MUST carry it, otherwise the exported model
        # silently serves the plain regression while the trainer-path score includes the
        # decode (measured: lambda=0.5 -> ONNX vs torch 3.8e-01; lambda=0 -> 2.4e-07).
        self.k_head_flag = k_head
        self.k_lambda = k_lambda
        self.kmax = 16
        self.nt = 2 * self.kmax + 1
        if k_head:
            self.k_head = nn.Linear(hid, 2 * self.nt)
            self.cls_head = nn.Linear(hid, 4)
        self.register_buffer("ticks", torch.tensor(
            [0.12543313, 0.24132973, 0.46853608, 0.65686768]), persistent=False)
        self.register_buffer("kv", torch.arange(-16, 17).float(), persistent=False)
        self.register_buffer("norm_mean", None)
        self.register_buffer("norm_std", None)

    def set_norm(self, mean, std):
        self.norm_mean = torch.from_numpy(mean).reshape(1, 1, -1)
        self.norm_std = torch.from_numpy(std).reshape(1, 1, -1)

    def u_split(self, head_out, h_out):
        """(u+, u-) heads with the gamma gate applied to u+ only -> (...,2)."""
        up, um = head_out[..., 0:1], head_out[..., 1:2]
        if self.gamma is not None:
            up = torch.sigmoid(self.gamma(h_out)) * up
        return torch.cat([up, um], dim=-1)

    def u_compose(self, head_out, h_out):
        """Rotate the two heads back to (t0,t1): p0=(g*u+ + u-)/sqrt2, p1=(g*u+ - u-)/sqrt2."""
        u = self.u_split(head_out, h_out)
        up, um = u[..., 0:1], u[..., 1:2]
        return torch.cat([(up + um) * INV_SQRT2, (up - um) * INV_SQRT2], dim=-1)

    def forward(self, features, hidden_0, hidden_1):
        if self.feat is not None:
            features = self.feat(features)
        if self.norm_mean is not None:
            features = (features - self.norm_mean) / self.norm_std
        raw_tick = features
        if self.persist is not None:              # D2, after normalization
            features = self.persist(features)
        if self.feat_mix is not None:             # F2: after norm, before GRU
            features = self.feat_mix(features)
        if self.layers == 1:
            h = hidden_0
        else:
            h = torch.cat([hidden_0, hidden_1], dim=0)  # (layers,1,hid)
        if self.front is not None:
            p = self.front(features)
            out, hn = self.gru(p, h)
            pred = self.head_d(self.post_ln(torch.cat([out, p], dim=-1)))
            if self.layers == 1:
                return pred, hn[0:1], torch.zeros_like(hidden_1)
            return pred, hn[0:1], hn[1:2]
        out, hn = self.gru(features, h)             # out (1,1,hid), hn (layers,1,hid)
        if self.gate is not None:                   # N6: feature-axis gate
            out = out * self.gate(features)
        h_out = out                                 # gamma reads h_t, never the raw tick
        if self.head_raw:                           # D1: concatenate the raw tick
            out = torch.cat([out, raw_tick], dim=-1)
        pred = self.head(out)                       # (1,1,2)
        if self.u_basis:                            # TZ-6: heads are (u+,u-)
            pred = self.u_compose(pred, h_out)
        if getattr(self, "k_head_flag", False):     # 26.10: discrete decode in the graph
            th = (torch.softmax(self.cls_head(h_out), dim=-1) * self.ticks).sum(-1)
            lg = self.k_head(h_out)
            kk = self.kv.to(lg.device)
            e0 = (torch.softmax(lg[..., :self.nt], dim=-1) * kk).sum(-1)
            e1 = (torch.softmax(lg[..., self.nt:], dim=-1) * kk).sum(-1)
            yh = torch.stack([e0 * th, e1 * th], dim=-1)
            pred = pred + self.k_lambda * (yh - pred)
        if self.layers == 1:
            return pred, hn[0:1], torch.zeros_like(hidden_1)
        return pred, hn[0:1], hn[1:2]


class StatefulTwin(nn.Module):
    """Пошаговый ONNX-wrapper для twin-encoder (ОДИН шаг за вызов).

    Контракт платформы не меняется:
      features[1,1,112] + hidden_0[1,1,2*H] + hidden_1[1,1,H/2]
      -> prediction[1,1,2] + next_hidden_0[1,1,2*H] + next_hidden_1[1,1,H/2],
    где H = width (по умолчанию 96). В hidden_0 склеены два состояния
    одинаковой ширины (h0 GRU-i0 и joint GRU, оба (1,B,H)), а в hidden_1 —
    состояние узкого GRU i1 (1,B,H/2). Это биективная упаковка тройного
    состояния twin-encoder (h0, h1, h) в два буфера контракта; ничего не
    теряется и состояние передаётся без потерь между шагами.
    """
    def __init__(self, width=96, mode="twinenc"):
        super().__init__()
        self.width = int(width)
        self.mode = str(mode).lower()
        self.twin = TwinEncGRU(self.width, mode=self.mode)
        self.norm_mean = None
        self.norm_std = None

    def set_norm(self, mean, std):
        self.norm_mean = torch.from_numpy(mean).reshape(1, 1, -1).clone()
        self.norm_std = torch.from_numpy(std).reshape(1, 1, -1).clone()

    def forward(self, features, hidden_0, hidden_1):
        x = features
        if self.norm_mean is not None:
            x = (x - self.norm_mean) / self.norm_std
        w = self.width
        h0 = hidden_0[..., :w]
        hjoint = hidden_0[..., w:]
        h1 = hidden_1
        out, (nh0, nh1, nh) = self.twin(x, (h0, h1, hjoint))
        next_h0 = torch.cat([nh0, nh], dim=-1)
        next_h1 = nh1
        return out, next_h0, next_h1


class FeatureTransform(nn.Module):
    """Torch mirror of feat_eng.make_features (raw 112 -> N_DERIVED).

    Batched in training, single-row inside ONNX at inference. Mirrors the numpy
    implementation exactly so the checkpoint, the exported graph and the numpy
    reference in feat_eng all agree (validated by tmp/verify_torch_feats.py).

    Doing the transform INSIDE the model graph matters for timing: per-row
    numpy cost measured ~345 us/row (~216 min for the test set) versus ~8 min
    for the GRU itself. Inside ONNX Runtime the same math is fused C++.
    """

    def __init__(self):
        super().__init__()
        self.eps = float(EPS)

    def forward(self, x):                       # x: (B,T,112)
        f = x
        b0b, b0a = f[..., I0_BEST_BID], f[..., I0_BEST_ASK]
        b1b, b1a = f[..., I1_BEST_BID], f[..., I1_BEST_ASK]
        s0 = b0a - b0b
        s1 = b1a - b1b
        m0 = 0.5 * (b0a + b0b)
        m1 = 0.5 * (b1a + b1b)

        v0 = f[..., I0_V]
        v1 = f[..., I1_V]
        v0p = torch.clamp(v0, min=0).sum(dim=-1)
        v0n = torch.clamp(v0, max=0).sum(dim=-1)
        v1p = torch.clamp(v1, min=0).sum(dim=-1)
        v1n = torch.clamp(v1, max=0).sum(dim=-1)
        eps = self.eps
        imb0 = (v0p + v0n) / (v0p - v0n + eps)
        imb1 = (v1p + v1n) / (v1p - v1n + eps)

        dp0, dv0 = f[..., I0_DP], f[..., I0_DV]
        dp1, dv1 = f[..., I1_DP], f[..., I1_DV]

        cols = [
            b0b, b0a, s0, m0, s0 / (torch.abs(m0) + eps),
            b1b, b1a, s1, m1, s1 / (torch.abs(m1) + eps),
            m0 - m1, s0 - s1,
            v0.sum(dim=-1), v0.std(dim=-1, unbiased=False), v0p, v0n, imb0,
            v1.sum(dim=-1), v1.std(dim=-1, unbiased=False), v1p, v1n, imb1,
            imb0 - imb1,
            dp0.mean(dim=-1), dp0.std(dim=-1, unbiased=False),
            dv0.sum(dim=-1), dv0.max(dim=-1).values, (dv0 > 0).float().mean(dim=-1),
            dp1.mean(dim=-1), dp1.std(dim=-1, unbiased=False),
            dv1.sum(dim=-1), dv1.max(dim=-1).values, (dv1 > 0).float().mean(dim=-1),
        ]
        a = [f[..., j] for j in range(A_ALL.start, A_ALL.stop)]
        return torch.stack(cols + a, dim=-1)


class FullSeqModel(nn.Module):
    """Offline training wrapper: full (B,T,F) -> (B,T,2) plus final hidden."""

    def __init__(self, hid=96, layers=2, dropout=0.0, n_features=N_FEATURES,
                 use_eng=False, head_raw=False, head_mlp=False, persist_col=-1, feat_gate=False, feat_mix=False,
                 u_basis=False, gamma_gate=False, k_head=False, k_lambda=0.5, cell='gru',
                 hf_features=False, hf_causal=False, aux_mix=0.0, aux_lag=1,
                 slow_mix=0.0, slow_eps=1e-4, lag_attn=0.0, feat_drop="", front_dim=0):
        super().__init__()
        self.hid = hid
        self.layers = layers
        self.use_eng = use_eng
        self.head_raw = head_raw          # D1: head sees [h, raw tick] not just [h]
        self.persist_col = persist_col    # D2: append this raw column as an extra input
        self.feat_gate = feat_gate        # N6: gate the GRU output by the raw row
        self.u_basis = u_basis            # TZ-6
        self.gamma_gate = gamma_gate      # TZ-6
        self.feat = FeatureTransform() if use_eng else None
        self.persist = PersistFeat(persist_col) if persist_col >= 0 else None
        self.n_features = N_DERIVED_ENG if use_eng else n_features
        if self.persist is not None:
            self.n_features += 1
        # FRONT-END (David Alami, 0.6645): Linear(F->front_dim) + GELU + LayerNorm,
        # GRU over the projection, then LayerNorm(concat[gru_out, proj]) + MLP head.
        # Off by default: front_dim=0 keeps every legacy checkpoint bit-identical.
        self.front_dim = int(front_dim)
        self.front = None
        self.post_ln = None
        self.head_d = None
        self.cell = cell
        if cell in ("twinenc", "twin"):
            # Новый тип ячейки: портированный twin-encoder (src/twin_arch.py).
            # Собственный модуль с собственным head; GRU-обёртки FullSeqModel не
            # используются, чтобы их инициализация/RNG-поток не пересекались с
            # legacy-чекпойнтами. width = hid, i1 width = hid//2, joint = hid.
            self.twin = TwinEncGRU(hid, mode=cell)
            self.gru = None
            self.rnn = None
        elif cell == 'lstm':
            # LSTM: иная динамика состояния (h, c) => потенциально иная корреляция
            # с записью. Создаётся ТОЛЬКО при явном флаге, чтобы при cell='gru'
            # RNG-поток и все legacy-чекпойнты остались побитово совместимыми.
            self.rnn = nn.LSTM(self.n_features, hid, num_layers=layers,
                               batch_first=True,
                               dropout=dropout if layers > 1 else 0.0)
            self.gru = None
        elif cell == 'slstm':
            # sLSTM (xLSTM, 11.11.2026): состояние (h,c,m,n), лог-нормализатор m и
            # нормализатор n. Скриптуем ячейку целиком — иначе пошаговый питон-цикл
            # на 20k шагов съедает больше времени, чем GPU. Создаётся только по флагу.
            self.slstm = torch.jit.script(SLSTM(self.n_features, hid, layers))
            self.gru = None
            self.rnn = None
        else:
            _gru_in = self.n_features
            if self.front_dim > 0:
                self.front = nn.Sequential(nn.Linear(self.n_features, self.front_dim),
                                           nn.GELU(), nn.LayerNorm(self.front_dim))
                _gru_in = self.front_dim
            self.gru = nn.GRU(_gru_in, hid, num_layers=layers, batch_first=True,
                              dropout=dropout if layers > 1 else 0.0)
        # H-F (hidden features): восстановлено из hf_e6 (state_dict + ONNX-граф).
        # hf_gen — GRU-16 со СВОИМ состоянием hidden_g; hf_head — Linear(16->8):
        # первые 4 выхода = scale (tanh), вторые 4 = shift; каждый повторяется 28 раз,
        # т.е. модулируют 112 нормализованных признаков блоками по 28.
        self.hf_features = hf_features
        # hf_causal (29.10): scale/shift КАЖДОГО шага из его СОБСТВЕННОГО hf_gen-выхода.
        # Legacy (hf_causal=False) берёт lin из ПОСЛЕДНЕГО шага сегмента (hg_out[:, -1])
        # => это LOOK-AHEAD: при обучении модуляция видит будущее, и пошаговый
        # (онлайн) инференс такого НЕ может повторить (измерено 28.10: hf_e6.onnx
        # расходится с torch-эталоном уже на шаге 0, Δ=0.0617; на платформе hf-пакет
        # дал 0.49290 против рекорда 0.659288, F203).
        self.hf_causal = hf_causal
        if hf_features:
            self.hf_gen = nn.GRU(self.n_features, 16, batch_first=True)
            self.hf_head = nn.Linear(16, 8)
            nn.init.zeros_(self.hf_head.weight)
            nn.init.zeros_(self.hf_head.bias)
            self._hg_state = None
        self.gate = FeatGate(hid, self.n_features) if feat_gate else None
        self.head = (nn.Sequential(nn.Linear(hid + N_FEATURES, hid), nn.ReLU(),
                                   nn.Linear(hid, 2))
                     if head_mlp else
                     nn.Linear(hid + (N_FEATURES if head_raw else 0), 2))
        assert head_raw or not head_mlp, "head_mlp requires head_raw"
        self.head_mlp = head_mlp
        if self.front_dim > 0:
            self.post_ln = nn.LayerNorm(hid + self.front_dim)
            self.head_d = nn.Sequential(nn.Linear(hid + self.front_dim, self.front_dim),
                                        nn.GELU(), nn.Linear(self.front_dim, 2))
        # F2: after head (RNG-stream order preserved).
        self.feat_mix = FeatMix(self.n_features) if feat_mix else None
        self.gamma = nn.Linear(hid, 1) if gamma_gate else None      # TZ-6, zero-init
        if self.gamma is not None:
            nn.init.zeros_(self.gamma.weight)
            nn.init.zeros_(self.gamma.bias)
        # K-HEAD (26.10): the target is an INTEGER multiple of the class tick (F085/F089):
        # y = k*tick_c with tick 0.12543/0.24133/0.46854/0.65687 and k integer. A regression
        # on such a target shrinks: measured on the record dump, |pred|>=1.6 in 0.91% of
        # values while 17.26% of the target is there, and HALF of the c3 metric weight sits
        # on k=3 rows the model never reaches. A distributional head over signed k decoded
        # as E[k]*tick does not shrink toward the mean by construction. Post-hoc snapping of
        # an existing regression does NOT work (F087: -0.0230, t -6.28) -- the level has to
        # be learned, not recovered.
        # Modules are created ONLY when enabled, so the RNG stream (and every legacy
        # checkpoint's weights) is bit-identical when the flag is off.
        self.k_head_flag = k_head
        self.k_lambda = k_lambda
        self.kmax = 16
        self.nt = 2 * self.kmax + 1
        if k_head:
            self.k_head = nn.Linear(hid, 2 * self.nt)
            self.cls_head = nn.Linear(hid, 4)
            self.register_buffer('ticks', torch.tensor(
                [0.12543313, 0.24132973, 0.46853608, 0.65686768]), persistent=False)
            self.register_buffer('kv', torch.arange(-self.kmax, self.kmax + 1).float(),
                                 persistent=False)
        # P1 (05.11): вспомогательная голова САМОПРЕДСКАЗАНИЯ — по скрытому состоянию
        # предсказывает признаки следующего шага. Учится только на трейне и на инференсе
        # не используется: граф вывода не меняется, конвертер кладёт gru+head как обычно.
        self.aux_mix = float(aux_mix)
        self.aux_lag = int(aux_lag)
        if self.aux_mix > 0:
            self.aux_head = nn.Linear(hid, self.n_features)
        # P2 (05.11): ДВА ВРЕМЕНИ В ОДНОМ СОСТОЯНИИ. Медленное состояние — причинная
        # экспоненциальная свёртка h_t (веса alpha*beta^k, обрезанные по eps), голова видит
        # [h_fast, h_slow]. Свёртка причинная (падинг только слева) ⇒ без look-ahead.
        self.slow_mix = float(slow_mix)
        self.slow_k = 0
        if self.slow_mix > 0:
            import math as _m
            beta = 1.0 - self.slow_mix
            self.slow_k = max(1, int(_m.ceil(_m.log(float(slow_eps)) / _m.log(beta))))
            w = self.slow_mix * beta ** torch.arange(self.slow_k + 1, dtype=torch.float32)
            w = w / w.sum()                       # единичное усиление по постоянной
            self.register_buffer("slow_w", w.view(1, 1, -1).repeat(hid, 1, 1), persistent=False)
            if head_raw:
                raise SystemExit("--slow-mix несовместим с --head-raw (голова уже расширена)")
            self.head = nn.Linear(2 * hid, 2)
        # P4 (05.11): внимание по оси лагов — 4 группы по 28 признаков, скор на лаг.
        self.lag_attn = float(lag_attn)
        if self.lag_attn > 0:
            self.lag_w = nn.Parameter(torch.zeros(4, self.n_features // 4))
        # D-05 (05.11): МАСКА ГРУПП ПРИЗНАКОВ — арм видит только часть входов.
        # Зачем: смесь из пяти армов одного рецепта не даёт разнообразия; арм на другой
        # группе признаков структурно непохож. Упаковка бесплатна: обнулённые колонки W_ih
        # эквивалентны выбрасыванию входов, правок ядра не нужно.
        self.feat_drop = feat_drop or ""
        if self.feat_drop:
            gmap = {"i0p": (0, 28), "i0v": (28, 56), "i1p": (56, 84), "i1v": (84, 112)}
            m = torch.ones(1, 1, self.n_features)
            for g in [s.strip().lower() for s in self.feat_drop.split(",") if s.strip()]:
                if g not in gmap:
                    raise SystemExit(f"--feat-drop: неизвестная группа {g!r} (есть {list(gmap)})")
                a, b = gmap[g]
                m[0, 0, a:b] = 0.0
            if float(m.sum()) == 0:
                raise SystemExit("--feat-drop: все группы выброшены")
            self.register_buffer("feat_mask", m, persistent=False)

    def slow_join(self, h):
        """P2: [h_fast, h_slow] — медленное состояние как причинная EMA-свёртка по времени."""
        if self.slow_mix <= 0:
            return h
        x = h.transpose(1, 2)                                  # (B,hid,T)
        x = torch.nn.functional.pad(x, (self.slow_k, 0))       # только слева ⇒ без look-ahead
        s = torch.nn.functional.conv1d(x, self.slow_w, groups=self.hid)
        return torch.cat([h, s.transpose(1, 2)], dim=-1)

    def lag_gate(self, x):
        """P4: внимание по оси лагов внутри каждой из 4 групп по 28 признаков.

        Скор лага = линейная форма его же значения, softmax по 28 лагам группы, затем
        строка умножается на 28*a (единица при равномерном внимании). Ширина входа не
        меняется ⇒ ядру понадобится только softmax по блокам, без изменения размеров.
        """
        if self.lag_attn <= 0:
            return x
        b, t, f = x.shape
        xg = x.view(b, t, 4, f // 4)
        s = xg * self.lag_w.view(1, 1, 4, -1)                  # скор КАЖДОГО лага
        a = torch.softmax(s, dim=-1) * (f // 4)                # softmax по 28 лагам группы
        gate = 1.0 - self.lag_attn + self.lag_attn * a
        return (xg * gate).view(b, t, f)

    u_split = StatefulGRU.u_split
    u_compose = StatefulGRU.u_compose

    def hf_modulate(self, x, hg=None):
        """H-F: модуляция нормализованного входа per-feature (блоки по 28).

        x  — (B, T, F) уже нормализованный вход; hg — состояние GRU-16 (1, B, 16).
        Возвращает (x_mod, h_g_new). Блок 28: 4 scale + 4 shift на 112 признаков.
        """
        if not self.hf_features:
            return x, hg
        if self.n_features % 4 != 0:
            raise ValueError(f"H-F expects n_features % 4 == 0, got {self.n_features}")
        rep = self.n_features // 4
        hg_out, hg_new = self.hf_gen(x, hg)
        if self.hf_causal:
            # КАУЗАЛЬНАЯ модуляция: scale/shift каждого шага из его собственного состояния.
            lin = self.hf_head(hg_out)                          # (B, T, 8)
            scale = (1.0 + torch.tanh(lin[..., :4])).repeat_interleave(rep, dim=-1)
            shift = lin[..., 4:].repeat_interleave(rep, dim=-1)
            return x * scale + shift, hg_new
        lin = self.hf_head(hg_out[:, -1])                       # (B, 8) — LEGACY: LOOK-AHEAD
        # ВАЖНО: в hf_e6.onnx масштаб = 1 + tanh(lin[:4]) (val_68 = 1), т.е. мультипликативный
        # гейт «1 + tanh», а НЕ чистый tanh (проверено: div[0]*scale[0] == mul[0]).
        scale = (1.0 + torch.tanh(lin[:, :4])).repeat_interleave(rep, dim=-1)
        shift = lin[:, 4:].repeat_interleave(rep, dim=-1)
        x_mod = x * scale.unsqueeze(1) + shift.unsqueeze(1)
        return x_mod, hg_new

    # ---- K-HEAD helpers (26.10) -------------------------------------------------
    @staticmethod
    def _seq_tick(y):
        """Per-sequence tick = the smallest NON-ZERO |target| in that sequence.

        F089: the class is exactly the target quantum, and min nonzero |y| equals the
        class tick to ~1e-9 for all 1873 sequences. So the tick is recoverable from the
        targets during training without any class-label file, and a sequence with no
        non-zero row falls back to 1.0.
        """
        ay = y.abs()
        big = torch.where(ay > 0, ay, torch.full_like(ay, 1e9))
        t = big.amin(dim=(1, 2))
        return torch.where(t < 1e8, t, torch.ones_like(t))

    def k_aux(self, h_out):
        """(logits over signed k for both targets, tick_hat) -- tick from a class head."""
        pooled = h_out.mean(dim=1)                                  # (B, hid)
        tick_hat = (torch.softmax(self.cls_head(pooled), dim=-1) * self.ticks).sum(-1)
        return self.k_head(h_out), tick_hat                        # (B,T,2*nt), (B,)

    def k_decode(self, logits, tick_hat):
        """E[k]*tick per target -- the non-shrinking decode."""
        kk = self.kv.to(logits.device)
        e0 = (torch.softmax(logits[..., :self.nt], dim=-1) * kk).sum(-1)
        e1 = (torch.softmax(logits[..., self.nt:], dim=-1) * kk).sum(-1)
        th = tick_hat.unsqueeze(1)
        return torch.stack([e0 * th, e1 * th], dim=-1)

    def k_apply(self, pred, h_out, y=None):
        """Blend the k-decode into pred and (optionally) return the CE loss.

        One knob: lambda. pred + lambda*(E[k]*tick - pred). The CE target is the exact
        integer k = round(y/tick_seq), clipped to +-kmax.
        """
        logits, tick_hat = self.k_aux(h_out)
        self.aux_last = (logits.detach(), tick_hat.detach())   # diagnostics only
        yh = self.k_decode(logits, tick_hat)
        pred = pred + self.k_lambda * (yh - pred)
        ce = None
        if y is not None:
            t = self._seq_tick(y)
            idx = torch.round(y / t.view(-1, 1, 1)).long() + self.kmax
            idx = idx.clamp(0, 2 * self.kmax)
            ce0 = F.cross_entropy(logits[..., :self.nt].reshape(-1, self.nt),
                                  idx[..., 0].reshape(-1))
            ce1 = F.cross_entropy(logits[..., self.nt:].reshape(-1, self.nt),
                                  idx[..., 1].reshape(-1))
            ce = 0.5 * (ce0 + ce1)
        return pred, ce

    def forward(self, x):
        if self.feat is not None:
            x = self.feat(x)
        raw_tick = x
        if self.persist is not None:              # D2
            x = self.persist(x)
        if self.feat_mix is not None:             # F2
            x = self.feat_mix(x)
        if self.hf_features:
            x, self._hg_state = self.hf_modulate(x, getattr(self, '_hg_state', None))
        if self.cell in ("twinenc", "twin"):
            # Twin-encoder — автономная архитектура: возвращает (out (B,T,2), hidden-тройку).
            out, hn = self.twin(x)
            if not isinstance(hn, (tuple, list)):
                hn = (hn,)
            return out, hn
        if self.feat_drop:                        # D-05: арм видит только часть групп
            x = x * self.feat_mask
        if self.lag_attn > 0:                     # P4: внимание по оси лагов
            x = self.lag_gate(x)
        if self.cell == 'lstm':
            out, (hn, cn) = self.rnn(x)
            self.last_cn = cn          # для пошагового ONNX-экспорта
        elif self.cell == 'slstm':
            out, hn = self.slstm(x)
            self.last_state = hn       # 4-состояние (h,c,m,n), см. export_onnx_slstm
        elif self.front is not None:
            p = self.front(x)
            out, hn = self.gru(p)
            return self.head_d(self.post_ln(torch.cat([out, p], dim=-1))), hn
        else:
            out, hn = self.gru(x)
        if self.gate is not None:                 # N6: feature-axis gate
            out = out * self.gate(x)
        h_out = out
        if self.head_raw:                         # D1: concatenate the (normalized) tick
            out = torch.cat([out, raw_tick], dim=-1)
        head_out = self.head(self.slow_join(out))
        if self.u_basis:                          # TZ-6
            head_out = self.u_compose(head_out, h_out)
        if self.k_head_flag:                      # 26.10: learnable discrete decode
            head_out, _ = self.k_apply(head_out, h_out, None)
        return head_out, hn


_PF_CACHE = {}
_FD_CACHE = {}
_PF_LOCK = None


def _get_fd(path):
    """Cached read-only fd for fadvise calls (same file as the ParquetFile)."""
    global _PF_LOCK
    import threading
    if _PF_LOCK is None:
        _PF_LOCK = threading.Lock()
    key = (threading.get_ident(), path)
    with _PF_LOCK:
        fd = _FD_CACHE.get(key)
    if fd is None:
        fd = os.open(path, os.O_RDONLY)
        with _PF_LOCK:
            _FD_CACHE[key] = fd
    return fd


def drop_cache_range(pf, fd, g):
    """Drop page-cache pages of row group g after reading.

    The host (Windows, no pagefile, ~9 GB commit headroom) restarts WSL when
    the guest page cache grows unchecked while streaming ~29 GB/epoch through
    the 9p bridge. fadvise(DONTNEED) keeps guest cache bounded; pages get
    re-read from disk on the next epoch anyway.
    """
    try:
        rg = pf.metadata.row_group(g)
        start, end = None, 0
        for i in range(rg.num_columns):
            col = rg.column(i)
            dpo = getattr(col, "dictionary_page_offset", None)
            po = col.data_page_offset
            base = min(x for x in (dpo, po) if x is not None)
            start = base if start is None else min(start, base)
            end = max(end, po + col.total_compressed_size)
        if start is not None and end > start:
            start -= start % 4096
            os.posix_fadvise(fd, start, end - start, os.POSIX_FADV_DONTNEED)
    except Exception:
        pass


def mem_snapshot():
    """Guest memory figures for the training log (KB from /proc/meminfo)."""
    try:
        info = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                if k in ("MemAvailable", "MemFree", "Cached"):
                    info[k] = int(v.strip().split()[0]) // 1024  # MB
        return info
    except Exception:
        return {}



def pooled_wp_torch(pred, y, clip=METRIC_CLIP, eps=1e-8, row_w=None, unit=False):
    """GLOBAL (pooled) weighted Pearson - the EXACT platform metric over a batch.

    WHY (21.10 2026, measured): the platform scorer
    (utils.GlobalAccumulator / weighted_pearson) centres the correlation
    GLOBALLY over all masked rows. The trainer has NEVER had a global loss:
    all three of its losses (block_wp, block_spearman, and until today
    level_corr) are per-sequence. block_wp is exactly blind to the
    between-sequence LEVEL, which is ~65% of the score; level_corr fixes the
    level but then the within-sequence shape and the level trade off against
    each other (measured: rho_level +0.026 at the cost of rho_block -0.008).

    This function removes the conflict BY CONSTRUCTION: it optimises exactly
    what the platform measures, so there is no second, misaligned objective
    to trade against. The risk is the opposite one - the global correlation is
    a much weaker/noisier gradient (one scalar for the whole batch instead of
    one per sequence), so it usually needs to be BLENDED with the block term.

    pred, y: (B, T, 2) float tensors. Returns a scalar in [-1, 1]
    (mean over the two targets, exactly as the platform does).
    """
    p = pred.clamp(-clip, clip)
    yc = y.clamp(-clip, clip)
    bad = ~torch.isfinite(p) | ~torch.isfinite(yc)
    p = torch.where(bad, torch.zeros_like(p), p)
    yc = torch.where(bad, torch.zeros_like(yc), yc)
    w = torch.ones_like(yc) if unit else yc.abs()   # minimal: unit = allrows objective
    if row_w is not None and not unit:
        w = w * row_w.unsqueeze(-1)
    W = w.sum(dim=(0, 1))                       # (2,)
    my = (w * yc).sum(dim=(0, 1)) / W.clamp_min(eps)
    mp = (w * p).sum(dim=(0, 1)) / W.clamp_min(eps)
    ycen = yc - my.view(1, 1, -1)
    pcen = p - mp.view(1, 1, -1)
    cov = (w * ycen * pcen).sum(dim=(0, 1)) / W.clamp_min(eps)
    vy = (w * ycen * ycen).sum(dim=(0, 1)) / W.clamp_min(eps)
    vp = (w * pcen * pcen).sum(dim=(0, 1)) / W.clamp_min(eps)
    sy = vy.clamp_min(0.0).sqrt()
    sp = vp.clamp_min(0.0).sqrt()
    rho = torch.where((sy > 1e-8) & (sp > 1e-8),
                      cov / (sy * sp).clamp_min(1e-12),
                      torch.zeros_like(cov)).clamp(-1.0, 1.0)
    return rho.mean()

def block_spearman_torch(pred, y, clip=METRIC_CLIP, eps=1e-6, row_w=None, tau=0.05):
    """H2: weighted Spearman (rank) correlation over the block, differentiable.

    WHY: the metric weights rows by |clip(y)|, and Pearson on those weights is
    dominated by the |y|~2 tail while the metric's mass sits in |y| in [0.2, 1.5].
    A rank term asks only for the ORDER to be right, which is what a correlation
    metric ultimately rewards.

    DIFFERENTIABILITY: exact ranks are non-differentiable, so ranks are replaced by
    a soft rank: r_i = sum_j sigmoid((x_j - x_i)/tau). As tau -> 0 this becomes the
    integer rank; tau is the smoothness knob. Rank the TARGETS in the same way and
    correlate the two rank vectors (that is Spearman = Pearson of ranks).

    Weights: the same |clip(y)| * row_w as the Pearson term, so both terms speak
    about the same rows.
    """
    p = pred.clamp(-clip, clip)
    yc = y.clamp(-clip, clip)
    w = yc.abs()
    if row_w is not None:
        w = w * row_w.unsqueeze(-1)                 # (B,T) -> (B,T,1)

    def soft_rank(x, anchors=512, detach=False):
        # x: (B,T,C) -> soft ranks along T.
        # A pairwise (B,T,T,C) tensor is quadratic: at T=8000 that is 2 GB per batch
        # item on a 4 GB card, and autograd keeps every chunk alive so chunking alone
        # does not help. Instead the rank of each row is estimated against a random
        # SUBSET of `anchors` rows: r_i ~ (T/anchors) * sum_{j in anchors} sigmoid(...).
        # Memory becomes (B,T,anchors,C), i.e. T*K instead of T*T - a 16x saving at
        # T=8000, K=512. This is an unbiased estimator of the true soft rank up to a
        # scale factor, and a rank term only needs the ordering to be right.
        src = x.detach() if detach else x
        T = src.shape[1]
        if anchors >= T:
            idx = torch.arange(T, device=src.device)
            K = T
        else:
            idx = torch.randperm(T, device=src.device)[:anchors]
            K = anchors
        ref = src[:, idx, :].unsqueeze(1)             # (B,1,K,C)
        d = (ref - src.unsqueeze(2)) / tau            # (B,T,K,C)
        return torch.sigmoid(d).sum(dim=2) * (T / K)  # (B,T,C)

    rp = soft_rank(p, anchors=512, detach=False)
    ry = soft_rank(yc, anchors=512, detach=True)      # y is a constant here
    Wc = w.sum(dim=1) + eps
    my = (w * ry).sum(dim=1) / Wc
    mp = (w * rp).sum(dim=1) / Wc
    ycen = ry - my.unsqueeze(1)
    pcen = rp - mp.unsqueeze(1)
    cov = (w * ycen * pcen).sum(dim=1) / Wc
    vy = (w * ycen * ycen).sum(dim=1) / Wc
    vp = (w * pcen * pcen).sum(dim=1) / Wc
    rho = cov / ((vy + eps).sqrt() * (vp + eps).sqrt())
    return rho.clamp(-1.0, 1.0).mean()


def block_wp_torch(pred, y, clip=METRIC_CLIP, eps=1e-6, row_w=None, unit=False):
    """Metric-aligned loss proxy: mean block-wise weighted Pearson.

    Mirrors `utils.weighted_pearson` exactly (clip both sides to +-2, weights
    w=|clip(y)|, weighted centering), but computed per sequence (block) in the
    batch and per target, then averaged — this is what the leaderboard scores.
    Returns the mean correlation (higher = better); the caller negates it.

    Numerics: eps is added INSIDE the sqrt arguments so the denominator can
    never reach 0 (a bare sqrt(0) has infinite gradient -> NaN loss). The
    official scorer's degenerate cases (weightless block, constant prediction)
    are handled by the same eps, which pushes rho toward a finite small value
    instead of exactly 0 — acceptable inside a loss.

    pred, y: (B, T, 2) float tensors.
    """
    p = pred.clamp(-clip, clip)
    yc = y.clamp(-clip, clip)
    w = torch.ones_like(yc) if unit else yc.abs()   # minimal: unit = allrows objective
    if row_w is not None and not unit:
        w = w * row_w.unsqueeze(-1)                 # (B,T) -> (B,T,1) broadcast
    W = w.sum(dim=1)                                # (B,2)
    Wc = W + eps
    my = (w * yc).sum(dim=1) / Wc                   # (B,2)
    mp = (w * p).sum(dim=1) / Wc                    # (B,2)
    ycen = yc - my.unsqueeze(1)
    pcen = p - mp.unsqueeze(1)
    cov = (w * ycen * pcen).sum(dim=1) / Wc
    vy = (w * ycen * ycen).sum(dim=1) / Wc
    vp = (w * pcen * pcen).sum(dim=1) / Wc
    denom = ((vy + eps) * (vp + eps)).sqrt()         # never 0 -> finite grad
    rho = (cov / denom).clamp(-1.0, 1.0)
    return rho.mean()


def global_wp_torch(pred, y, clip=METRIC_CLIP, eps=1e-6, row_w=None, unit=False):
    """Pooled (global) weighted Pearson over ALL rows in the batch.

    WHY. `block_wp_torch` averages per-sequence correlations. The scored metric
    pools every row together, so the per-sequence version optimises a freedom
    the metric removed: a model can raise the block mean by fitting each
    sequence's own offset, while the global number stays flat. The dozken
    reference (public 0.6569) reports this as the change that matters.

    Identical numerics to `block_wp_torch` except the reduction is over
    (B,T) jointly instead of dim=1 -- same clip, same |y| weights, same eps
    inside the sqrt so the denominator never reaches 0.

    Returns a scalar (higher = better); the caller negates it.
    """
    p = pred.clamp(-clip, clip)
    yc = y.clamp(-clip, clip)
    w = torch.ones_like(yc) if unit else yc.abs()   # minimal: unit = allrows objective
    if row_w is not None and not unit:
        w = w * row_w.unsqueeze(-1)                 # (B,T) -> (B,T,1)
    W = w.sum(dim=(0, 1))                           # (2,) per target
    Wc = W + eps
    my = (w * yc).sum(dim=(0, 1)) / Wc
    mp = (w * p).sum(dim=(0, 1)) / Wc
    ycen = yc - my
    pcen = p - mp
    cov = (w * ycen * pcen).sum(dim=(0, 1)) / Wc
    vy = (w * ycen * ycen).sum(dim=(0, 1)) / Wc
    vp = (w * pcen * pcen).sum(dim=(0, 1)) / Wc
    denom = ((vy + eps) * (vp + eps)).sqrt()
    rho = (cov / denom).clamp(-1.0, 1.0)            # (2,)
    return rho.mean()


def segment_states(T, seg, reset_every):
    """Yield (start, end, reset) blocks for truncated BPTT with an optional state reset.

    reset_every > 0 zeroes the GRU hidden state at the start of every block whose
    offset is a multiple of reset_every, instead of carrying it across segments.
    The dozken reference measures zero-state beating accumulated state after ~10k
    rows (reset2k/reset4k global 0.6864/0.6823), so the carry we currently do is
    the thing to ablate.

    The first block always starts clean: there is no state yet.

    Block boundaries stay on the seg grid, so a reset_every that is not a
    multiple of seg only lands where the two grids coincide.
    """
    if seg <= 0:
        seg = T
    blocks = []
    for start in range(0, T, seg):
        end = min(start + seg, T)
        if start == 0:
            reset = True
        elif reset_every and reset_every > 0:
            reset = (start % reset_every) == 0
        else:
            reset = False
        blocks.append((start, end, reset))
    return blocks


def block_wp_dnpl_torch(pred, y, clip=METRIC_CLIP, eps=1e-6, row_w=None):
    """Dispersion-Normalised PCC (DNPL): rho, scaled by sg(sigma_pred / sigma_y).

    WHY. Pearson rho is invariant to the scale of `pred`, so a model can raise it
    by SHRINKING its output toward the mean — exactly the pathology we measured:
    corr_mix 1.0 gives official 0.711438 but allrows 0.153724 (F273/F282), and the
    platform punishes that drift (F366: corr_mix 0.60/0.65 scored BELOW the record
    although local official was higher). The MSE anchor only partially prevents it.

    FIX (from the LOB literature, dispersion-normalised PCC): multiply the rho
    term by the stop-gradient ratio sigma_pred/sigma_y. A shrunken prediction has
    sigma_pred << sigma_y, so its reward collapses; no extra gradient path is
    introduced because the ratio is detached, so rho itself is unchanged.

    Returns the mean of rho * clamp(sg(sigma_pred/sigma_y), 0, 1).
    """
    p = pred.clamp(-clip, clip)
    yc = y.clamp(-clip, clip)
    w = yc.abs()
    if row_w is not None:
        w = w * row_w.unsqueeze(-1)
    W = w.sum(dim=1)
    Wc = W + eps
    my = (w * yc).sum(dim=1) / Wc
    mp = (w * p).sum(dim=1) / Wc
    ycen = yc - my.unsqueeze(1)
    pcen = p - mp.unsqueeze(1)
    cov = (w * ycen * pcen).sum(dim=1) / Wc
    vy = (w * ycen * ycen).sum(dim=1) / Wc
    vp = (w * pcen * pcen).sum(dim=1) / Wc
    denom = ((vy + eps) * (vp + eps)).sqrt()
    rho = (cov / denom).clamp(-1.0, 1.0)
    ratio = ((vp + eps) / (vy + eps)).sqrt().detach().clamp(0.0, 1.0)
    return (rho * ratio).mean()


def block_wp_chunked_torch(pred, y, chunk=0, clip=METRIC_CLIP, row_w=None, unit=False):
    """Средняя блочная корреляция, посчитанная ОКНАМИ по `chunk` строк.

    Зачем. Публично опубликованный сильный стек этой задачи пересчитывает Pearson каждые 1000 строк
    батча и заявляет ~0.6645 против наших 0.65948 (F381). У нас корреляционный член считается по всей
    последовательности сразу, и гипотеза состоит в том, что оконный пересчёт меняет профиль градиента:
    короткие окна дают более «локальный» сигнал и меньше размывания по длинной последовательности.

    chunk=0 или chunk>=T — тождественно обычному расчёту (проверяется тестом).
    Окна не перекрываются; хвост, не попавший в целое число окон, отбрасывается — так же поступает
    оригинальный приём (пересчёт по мере поступления строк).
    """
    if not chunk or chunk <= 0:
        return block_wp_torch(pred, y, clip=clip, row_w=row_w)
    B, T, C = pred.shape
    n = T // chunk
    if n < 2:
        return block_wp_torch(pred, y, clip=clip, row_w=row_w)
    pr = pred[:, :n * chunk].reshape(B * n, chunk, C)
    yr = y[:, :n * chunk].reshape(B * n, chunk, C)
    rw = None if row_w is None else row_w[:, :n * chunk].reshape(B * n, chunk)
    return block_wp_torch(pr, yr, clip=clip, row_w=rw)


def pooled_wp_chunked_torch(pred, y, chunk=1000, clip=METRIC_CLIP, eps=1e-8, row_w=None, unit=False):
    """Pooled weighted Pearson per chunk-of-batch, averaged over chunks.

    WHY. The public 0.6645 stack recomputes Pearson "every 1000 rows of the batch"
    (F381). That is a POOLED object across the batch's sequences, not the per-sequence
    block correlation of block_wp_chunked_torch: the per-sequence version is blind to
    the between-sequence level and lets the model drift (measured 07.11 on pub256w:
    block 0.683 +0.021 vs control, global 0.471 -0.242). This function centres over
    ALL rows of a chunk (all sequences), so the level is inside the objective.

    chunk=0 or chunk>=T is identical to pooled_wp_torch over the whole window.
    The tail remainder is dropped, matching the streaming 1000-row recompute.
    """
    if not chunk or chunk <= 0 or chunk >= pred.shape[1]:
        return pooled_wp_torch(pred, y, clip=clip, eps=eps, row_w=row_w)
    B, T, C = pred.shape
    n = T // chunk
    rhos = []
    for c in range(n):
        sl = slice(c * chunk, (c + 1) * chunk)
        p = pred[:, sl, :].reshape(1, B * chunk, C)
        yy = y[:, sl, :].reshape(1, B * chunk, C)
        rw = None if row_w is None else row_w[:, sl].reshape(1, B * chunk)
        rhos.append(pooled_wp_torch(p, yy, clip=clip, eps=eps, row_w=rw))
    return torch.stack(rhos).mean()


def target_delta_scale(y, mask=None):
    """δ_s = min |y| > 0 over the scored rows of each sequence; shape (B,1,1).

    The target is quantised (y = k·δ_s, four instruments). Replacing the target by
    k = y/δ_s makes the loss dimensionless and the model output a tick count, which
    inference multiplies by δ_s (--target-delta, plan v3 B-A). The prediction is NOT
    rescaled in training: u is compared with k directly.
    Sequences without a non-zero tick fall back to 1.0.
    """
    a = y.abs()
    if mask is not None:
        a = a * mask.unsqueeze(-1)
    a = torch.where(a > 0, a, torch.full_like(a, float("inf")))
    d = a.amin(dim=(1, 2), keepdim=True)
    return torch.where(torch.isfinite(d), d, torch.ones_like(d))


def corr_term_torch(pred, y, horizon="seq", pool_chunk=0, chunk=0, row_w=None, unit=False):
    """Correlation term for one training window, chosen by --corr-horizon.

    'global' is the POOLED object over all sequences of the batch — the exact
    platform metric (F381). Before 13.11 the 'global' branch was an alias of
    'seq' (per-sequence block term), which made the v3 plan's A arm a duplicate.
    pool_chunk > 0 keeps the public-stack chunking (1000 rows of the batch).
    """
    if horizon == "global" or pool_chunk > 0:
        return pooled_wp_chunked_torch(pred, y, chunk=pool_chunk, row_w=row_w, unit=unit)
    return block_wp_chunked_torch(pred, y, chunk=chunk, row_w=row_w, unit=unit)


_PHAT_W = None
_PHAT_B = None
_PHAT_NF = 0
_PHAT_W_NP = None
_PHAT_B_NP = None




from atten_loss import atten_shrink_torch, atten_diag_torch, noise_excess_torch


def scale_shrink_torch(pred, y, clip=METRIC_CLIP, eps=1e-4, row_w=None):
    """Penalty for OUTPUT SHRINKAGE, per sequence.

    WHY (21.10 2026, measured on the champion npz, 120 seq):
        slope of the per-sequence regression y ~ pred:
            class 0 (d=0.12543): 0.6307   std(y)=0.8719  std(pred)=0.9205
            class 1 (d=0.24133): 0.6974   std(y)=0.8883  std(pred)=0.8705
            class 2 (d=0.46854): 0.6501   std(y)=0.8911  std(pred)=0.8980
            class 3 (d=0.65687): 0.6449   std(y)=0.8165  std(pred)=0.5828  <- HALVED
        i.e. in the light classes the output scale is normal, but in the HARD
        class the model shrinks its output almost twofold. That is why hard_r is
        0.4714 while the others are 0.73-0.77.

    The oracle fix (align the output mean to the target mean) lifts hard_r to
    0.5306 (+0.058) - but it needs the target, so it is illegal at inference,
    and a global rescale does nothing (Pearson is invariant to scale). So the
    shrinkage has to be penalised DURING TRAINING.

    The term below is the direct expression of the defect: for every sequence
    it compares the weighted std of the prediction with the weighted std of the
    target and penalises only the case where the prediction is TOO NARROW:

        penalty = relu( 1 - std(pred_seq) / (std(y_seq) + eps) )

    It is zero when the prediction is at least as wide as the target, so it
    cannot push the model to over-disperse; it is 1 at total collapse.

    pred, y: (B, T, 2). Returns a scalar in [0, 1].
    """
    p = pred.clamp(-clip, clip)
    yc = y.clamp(-clip, clip)
    bad = ~torch.isfinite(p) | ~torch.isfinite(yc)
    p = torch.where(bad, torch.zeros_like(p), p)
    yc = torch.where(bad, torch.zeros_like(yc), yc)
    w = yc.abs()
    if row_w is not None:
        w = w * row_w.unsqueeze(-1)
    # per (sequence, target): weighted std
    sw = w.sum(dim=1).clamp_min(eps)                     # (B, 2)
    mean_p = (w * p).sum(dim=1) / sw
    mean_y = (w * yc).sum(dim=1) / sw
    var_p = (w * (p - mean_p.unsqueeze(1)) ** 2).sum(dim=1) / sw
    var_y = (w * (yc - mean_y.unsqueeze(1)) ** 2).sum(dim=1) / sw
    std_p = var_p.clamp_min(0.0).sqrt()
    std_y = var_y.clamp_min(0.0).sqrt()
    ratio = std_p / (std_y + eps)
    pen = torch.clamp(1.0 - ratio, min=0.0)
    # only sequences that carry signal
    valid = (sw > eps).float()
    return (pen * valid).sum() / valid.sum().clamp_min(1.0)

def shock_wp_torch(pred, y, clip=METRIC_CLIP, eps=1e-6, row_w=None, rho_ar1=0.93,
                   already_innovative=False):
    """A2 (24.10): SHOCK-anchored block WP — correlation of the INNOVATIONS.

    The publishable reserve (measured 19.10): the 'shock' w = y_t − 0.93·y_{t-1}
    is predicted from the features at r≈0.42 while the model's own output
    tracks it at only r≈0.275. block_wp_torch is blind to it: innovations are
    a small fraction of the level signal and the metric centres + weights
    them away (std_shock/std_y = 0.53, and the weight w=|y| re-focuses the
    loss on levels). This term correlates P_shock := P_t − 0.93·P_{t-1}
    against Y_shock := Y_t − 0.93·Y_{t-1} with SHOCK weights |Y_shock| so
    the gradient goes exactly into what the level-anchored metric misses.

    row_w (if given) must be ALREADY ALIGNED to pred's rows (the caller
    pre-crops; no inner crop here — the earlier double-crop bug threw
    IndexError at e0 step 0). already_innovative=True skips the
    differencing when pred/y are ALREADY innovations (unused today).
    One field vs control; inert at 0.0.
    """
    p = pred.clamp(-clip, clip)
    yc = y.clamp(-clip, clip)
    if already_innovative:
        ps, ys = p, yc
    else:
        ps = p[:, 1:, :] - rho_ar1 * p[:, :-1, :]
        ys = yc[:, 1:, :] - rho_ar1 * yc[:, :-1, :]
    if row_w is not None:
        w = ys.abs().clamp_min(eps) * row_w[:, 1:].unsqueeze(-1)     # caller full-aligned
    else:
        w = ys.abs().clamp_min(eps)
    W = w.sum(dim=1)
    Wc = W + eps
    my = (w * ys).sum(dim=1) / Wc
    mp = (w * ps).sum(dim=1) / Wc
    ycen = ys - my.unsqueeze(1)
    pcen = ps - mp.unsqueeze(1)
    cov = (w * ycen * pcen).sum(dim=1) / Wc
    vy = (w * ycen * ycen).sum(dim=1) / Wc
    vp = (w * pcen * pcen).sum(dim=1) / Wc
    denom = (vy * vp).clamp_min(eps).sqrt()
    return (cov / denom).mean()


def level_corr_torch(pred, y, clip=METRIC_CLIP, eps=1e-6, row_w=None):
    """Weighted correlation of the per-SEQUENCE LEVELS of pred and y, over the batch.

    WHY (21.10 2026, measured): `block_wp_torch` centres INSIDE each sequence,
    so it is EXACTLY invariant to a per-sequence shift of the prediction
    (measured: delta = 0.00e+00 at shift sd 0.2, while the platform metric on
    the same shift moves by +0.198). The platform metric (utils.weighted_pearson)
    centres GLOBALLY, so the between-sequence LEVEL is part of the score.
    This term is the part the current loss cannot see.

    Level of a sequence = the |y|-weighted mean of y (resp. pred) over that
    sequence's rows - exactly the convention the platform uses.

    pred, y: (B, T, 2) float tensors. Returns a scalar in [-1, 1].
    """
    p = pred.clamp(-clip, clip)
    yc = y.clamp(-clip, clip)
    bad = ~torch.isfinite(p) | ~torch.isfinite(yc)
    p = torch.where(bad, torch.zeros_like(p), p)
    yc = torch.where(bad, torch.zeros_like(yc), yc)
    w = yc.abs()
    if row_w is not None:
        w = w * row_w.unsqueeze(-1)
    W = w.sum(dim=1)
    my = (w * yc).sum(dim=1) / W.clamp_min(eps)
    mp = (w * p).sum(dim=1) / W.clamp_min(eps)
    W64 = W.double(); my64 = my.double(); mp64 = mp.double()
    Wm = W64.sum(dim=0)
    gmy = (W64 * my64).sum(dim=0) / Wm.clamp_min(eps)
    gmp = (W64 * mp64).sum(dim=0) / Wm.clamp_min(eps)
    dy = my64 - gmy
    dp = mp64 - gmp
    cov = (W64 * dy * dp).sum(dim=0) / Wm.clamp_min(eps)
    vy = (W64 * dy * dy).sum(dim=0) / Wm.clamp_min(eps)
    vp = (W64 * dp * dp).sum(dim=0) / Wm.clamp_min(eps)
    sy = vy.clamp_min(0.0).sqrt()
    sp = vp.clamp_min(0.0).sqrt()
    rho = torch.where((sy > 1e-8) & (sp > 1e-8),
                      cov / (sy * sp).clamp_min(1e-12),
                      torch.zeros_like(cov)).clamp(-1.0, 1.0)
    return rho.mean()

def _phat_hard_weights(p, select, threshold=None, topk=None):
    """PROXY-MASK class (A2, 22.10): HARD row selection 1/0 computed from p_hat only.

    Soft p_hat^gamma dilutes the gradient concentration that makes the exact
    is_scored mask the largest measured effect of the project (+0.0131, t=7.28,
    docs/RESEARCH_00_MASK_HEADLINE.md): soft weights give only +0.007, t=1.0.
    Hard selection keeps the concentration; because p_hat is a proxy (AUC 0.7563,
    tmp/mask_logit_coeffs.json) the rows it picks are "similar-but-not-exactly"
    the platform-scored ones, which is the bet for better transfer than the
    exact-mask recipe (isolation: docs/MA2_MASK_ISOLATION_VERDICT_20261022.md).

    select='threshold': weight = 1{p_hat > threshold}   (global threshold)
    select='topk':      weight = 1 at the K highest-p_hat rows per sequence
                        (dose fixed per sequence; robust to train/valid
                        distribution shift of p_hat, mean 0.148 train vs 0.104
                        valid, docs/RESEARCH_00_MASK_HEADLINE.md §2.4)

    Returns (weights float (B,T) in {0,1}, selected fraction).
    """
    if select == "threshold":
        hard = (p > float(threshold)).float()
    elif select == "topk":
        k = min(int(topk), p.shape[-1])
        kth = torch.topk(p, k, dim=-1).values[..., -1:].clamp_min(0.0)  # K-th value/seq
        hard = (p >= kth).float()
    else:
        raise ValueError(f"unknown phat_select: {select}")
    frac = float(hard.mean().item())
    return hard, frac


def _phat_coeffs_path(path):
    """Resolve the coeffs JSON: WSL default (/mnt/d/...) falls back to the
    repo-local tmp/ copy so selftest and Windows runs stay 0-arg."""
    import os
    if os.path.exists(path):
        return path
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "tmp", "mask_logit_coeffs.json")
    if os.path.exists(here):
        return here
    return path


def _load_phat_coeffs(path):
    """Load the logistic for P(is_scored): w_raw (8 or 112) + b_raw.

    JSON produced by tmp/mask_probe5.py (8 features, 'a-only') or
    tmp/fit_phat112.py (112 features, 'full'). The optional "n_features" key
    selects the input slice; absent = legacy 8-feature behaviour.
    """
    import json
    with open(path) as fh:
        c = json.load(fh)
    w = torch.tensor(c["w_raw"], dtype=torch.float32)
    b = float(c["b_raw"])
    nf = int(c.get("n_features", len(w)))
    return w, b, nf


def rolling_z_torch(x, W, eps=1e-5):
    """Causal rolling z-score: mean/std over the last W rows INCLUDING the current one.

    R2 line B (26.10): the recipe standardises the input with ONE global feat_stats
    baked into the graph, which cannot remove a per-sequence level drift (the first
    88 of the 112 features are absolute ladder prices/volumes). This normalises each
    row by its OWN recent window instead, strictly causally (x_t is available at
    prediction time, so including it is legal).

    x: (B,T,F) raw. Returns the same shape, clamped to +-8 so one degenerate window
    cannot blow up the GRU. There is NO numpy mirror in the repo today: the stock
    `solution.py` (`tools/ens_mhybrid_m4_solution_stock.py`) feeds RAW rows, and a roll-norm
    checkpoint exported through it scores garbage (measured 26.10: r2_rollnorm1024_e0 0.5387
    raw vs 0.6922 with this path). A mirror plus an equivalence test is a PREREQUISITE for
    ever shipping a roll-norm cell; until then `tools/export_tz6_member.py` REFUSES such
    checkpoints and scoring goes through `tools_class/score_ckpt_direct.py --roll-norm W`.
    """
    B, T, F = x.shape
    x64 = x.double()
    cs = torch.cumsum(x64, dim=1)
    cs2 = torch.cumsum(x64 * x64, dim=1)
    idx = torch.arange(1, T + 1, device=x.device, dtype=torch.int64)
    lo = (idx - W).clamp_min(0)                                   # (T,) rows (lo, t] in window
    take = (lo - 1).clamp_min(0).long()
    zero = torch.zeros(1, 1, F, device=x.device, dtype=torch.float64)
    prev1 = torch.where((lo == 0).view(1, -1, 1), zero, cs[:, take, :])
    prev2 = torch.where((lo == 0).view(1, -1, 1), zero, cs2[:, take, :])
    S1 = cs - prev1
    S2 = cs2 - prev2
    n = (idx - lo).clamp_min(1).double().view(1, -1, 1)
    mean = S1 / n
    var = (S2 / n - mean * mean).clamp_min(0.0)
    z = (x64 - mean) / torch.sqrt(var + eps)
    return z.float().clamp(-8.0, 8.0)


def _seg_bounds(t_len_, first_, seg_, reset_):
    """Segment (start, end) pairs, split at every reset boundary.

    With reset_ <= 0 this is exactly `range(first_, t_len_, seg_)`. With a
    reset grid finer than seg_, each block is cut at the boundaries inside
    it, so reset=2000 and reset=4000 actually differ -- otherwise every
    seg boundary is already a multiple of both and the arms are identical.
    """
    if reset_ <= 0:
        for st in range(first_, t_len_, seg_):
            yield st, min(st + seg_, t_len_)
        return
    st = first_
    while st < t_len_:
        blk_end = min(st + seg_, t_len_)
        nxt = ((st // reset_) + 1) * reset_
        end_st = min(blk_end, nxt)
        yield st, end_st
        st = end_st


def train_step(model, opt, feats, targs, args, device, mean_t=None, std_t=None,
               teachers=None, kd_alpha=0.0, row_mask=None):
    # A1 hard-carrier dose: per-row-group hard-kappa map, set by main() from
    # --row-weight-hard-kappa + the classmap. Empty dict = inert (identical loss).
    """One optimizer step over a batch of sequences.

    With --seg-len > 0 uses truncated backprop through time: the sequence is
    split into segments of seg_len steps, the hidden state is detached and
    carried forward, and gradients accumulate over segments before one
    optimizer step (memory ~ 20k/seg_len times lower).
    With --weighted-loss the MSE is weighted by |clipped target|, matching
    the metric's weights.
    With --corr-loss the loss becomes a metric-aligned block-wise weighted
    Pearson term (--corr-mix blends it with the MSE term; mix=1.0 = pure).
    With teachers (knowledge distillation) the loss adds kd_alpha * MSE to the
    teachers' mean prediction (teachers must use the same standardization).
    """
    atten_diag_acc = {}
    atten_diag_n = [0]
    x_raw = torch.from_numpy(feats).to(device)      # (B,T,112) RAW
    x = x_raw
    if getattr(args, "roll_norm", 0):
        x = rolling_z_torch(x_raw, int(args.roll_norm))   # R2: causal window replaces feat_stats
    elif mean_t is not None:
        x = (x - mean_t) / std_t
    # input dropout (training only; the exported ONNX never sees it)
    if model.training and getattr(args, "input_dropout", 0.0) > 0:
        _p = float(args.input_dropout)
        _keep = (torch.rand_like(x) >= _p).to(x.dtype) / (1.0 - _p)
        x = x * _keep
    row_w = None
    _PHAT_NF = globals().get("_PHAT_NF", 0)
    if row_mask is not None:
        rm = torch.from_numpy(row_mask).to(device).float() if not torch.is_tensor(row_mask) \
            else row_mask.to(device).float()
        if getattr(args, "row_weight_abs", False):
            rm = rm.abs()
        elif bool((rm < 0).any()):
            raise SystemExit(
                "--row-weight-col yielded NEGATIVE weights (min %.3f). Weights must be >= 0; "
                "pass --row-weight-abs to use |col| (e.g. a signed target column). "
                "12.11.2026: 0.32 vs 0.71 — signed weights silently destroyed training."
                % float(rm.min()))
        row_w = rm.clamp_min(getattr(args, "row_weight_floor", 0.0)) \
            ** getattr(args, "row_weight_gamma", 1.0)
    elif getattr(args, "row_weight_mode", "none") == "phat":
        if _PHAT_NF == 112:
            z = x_raw @ _PHAT_W                       # (B,T,112) -> (B,T)
        else:
            z = x_raw[..., 104:112] @ _PHAT_W         # legacy: last 8 features
        z = z + _PHAT_B
        p = torch.sigmoid(z)                 # (B,T)
        _sel = getattr(args, "phat_select", "none")
        if _sel == "none":
            row_w = p.clamp_min(getattr(args, "phat_floor", 0.02)) \
                ** getattr(args, "phat_gamma", 2.0)       # (B,T)
        else:                                # PROXY-MASK: hard 1/0 selection
            row_w, _sel_frac = _phat_hard_weights(
                p, _sel, getattr(args, "phat_threshold", None),
                getattr(args, "phat_topk", None))
    # BLOCK-EQ (03.11.2026): метрика усредняет БЛОКИ, а лосс — СТРОКИ, поэтому последовательности
    # с редкой маской недопредставлены. Измерено (F285): счёт блока падает с ростом плотности
    # маски (r=-0.362), и худшие блоки имеют mask_frac 0.149 против 0.069 у лучших. Флаг нормирует
    # веса строк на суммарный вес последовательности, сохраняя средний масштаб (x T).
    # OFF => путь вычислений побайтово прежний (блок не исполняется).
    if getattr(args, "row_weight_blockeq", False) and row_w is not None:
        _S = row_w.sum(dim=1, keepdim=True).clamp_min(1e-9)
        row_w = row_w / _S * row_w.shape[1]
    # CHANGE-BOOST (10.11.2026): метрика различает модели только на строках СМЕНЫ ступени.
    # Измерено на полном valid нашим прибором (docs/LAG_BASELINE_20261110.md): 92.1% scored-строк —
    # удержание (лаг-1 там 0.99977, различать нечего), все различия живут на 7.87% строк смены,
    # где наша модель даёт 0.825 против 0.850 у оракула. Лосс взвешен |is_scored| и этого не знает.
    # OFF (0.0) => путь вычислений побайтово прежний.
    if getattr(args, "change_boost", 0.0) > 0:
        _y0 = np.clip(np.asarray(targs)[..., 0], -METRIC_CLIP, METRIC_CLIP).astype(np.float32)
        _ch = np.zeros(_y0.shape, dtype=np.float32)
        _ch[:, 1:] = (np.abs(_y0[:, 1:] - _y0[:, :-1]) > 0).astype(np.float32)
        _chw = torch.from_numpy(1.0 + float(args.change_boost) * _ch).to(device)
        # 10.11.2026 BUGFIX: с --row-weight-col is_scored все train-строки получают вес 0
        # (is_scored=True только у valid-групп: проверено, 0 из 260k сэмплированных train-строк),
        # поэтому умножение на _chw не давало НИЧЕГО (mean_w=1.28, а rw_mean=0.0 и loss байт-в-байт
        # как без флага). Если базовый вес нулевой - берём единицы, иначе буст умножается на ноль.
        if row_w is not None and float(row_w.mean()) == 0.0:
            row_w = _chw
        else:
            row_w = _chw if row_w is None else row_w * _chw
        if not globals().get("_CHANGE_LOGGED", False):
            globals()["_CHANGE_LOGGED"] = True
            print(f"[change-boost] lambda={args.change_boost} mean_w={float(_chw.mean()):.4f} "
                  f"share_changed={float(_ch.mean()):.4f}", flush=True)
    # FAIL-FAST (10.11.2026): нулевой средний вес строки означает, что лосс вырожден -
    # так случилось, когда --row-weight-col указывал на is_scored, а в train-группах он всегда False
    # (проверено: 0 из 260k строк). Прогон при этом выглядел здоровым: лосс падал, rho стоял на 0.0000,
    # модель сохранялась. Лучше упасть на первом батче, чем получить тихо неверную модель.
    if row_w is not None and float(row_w.mean()) == 0.0:
        raise SystemExit(
            "[train] FATAL: средний вес строки = 0. Лосс вырожден (корреляционный член и MSE "
            "обнулены). Проверь --row-weight-col: скорее всего колонка нулевая на этом участке "
            "данных (например is_scored истинно только в valid-группах). "
            "Для train нужен --row-weight-hybrid <N> --row-weight-mask-file ... либо откажись "
            "от весов строк вовсе."
        )
    y = torch.clamp(torch.from_numpy(targs).to(device), -METRIC_CLIP, METRIC_CLIP)
    y_raw_abs = y.abs()          # metric weights |clip(y,2)|, before any target transform
    # C1' (MA2 20.10): target-side standardization (ScaleIn, 2610.07324).
    # ОДНА переменная: потеря считается на y' = (y - m_s)/sigma_s, где m_s/sigma_s —
    # взвешенные (w=|y|) mean/std по SCORED-строкам КАЖДОЙ последовательности (f64),
    # предсказание нормируется тем же (m_s, sigma_s) ДО лосса.
    # Механизм: обратное преобразование умножает градиент на sigma_s ⇒ масштаб группы
    # перестаёт быть скрытым importance weight (4 носителя с разным |y|).
    # Флаг OFF => путь вычислений побайтово прежний (анкор step0 = 1.427292).
    tgt_ms = tgt_sd = None
    tgt_delta = None
    if getattr(args, 'target_standardize', False):
        _y64 = y.double()
        if row_mask is not None:
            _rm = torch.from_numpy(row_mask).to(device).double() if not torch.is_tensor(row_mask) \
                else row_mask.to(device).double()
            _mask = (_rm > 0).unsqueeze(-1).double()
        else:
            _mask = torch.ones_like(_y64)
        _w = _y64.abs() * _mask
        _W = _w.sum(dim=1, keepdim=True).clamp_min(1e-9)
        _m = (_w * _y64).sum(dim=1, keepdim=True) / _W
        _v = (_w * (_y64 - _m) ** 2).sum(dim=1, keepdim=True) / _W
        _sd = _v.clamp_min(1e-12).sqrt()
        tgt_ms, tgt_sd = _m.float(), _sd.float()
        y = ((_y64 - _m) / _sd).float()
    if getattr(args, 'target_delta', False):
        _yd = y.double()
        if row_mask is not None:
            _rmd = (torch.from_numpy(row_mask).to(device) if not torch.is_tensor(row_mask)
                    else row_mask.to(device))
            _mkd = _rmd > 0
        else:
            _mkd = torch.ones(y.shape[:2], dtype=torch.bool, device=y.device)
        tgt_delta = target_delta_scale(_yd, _mkd).float()
        y = (_yd / tgt_delta.double()).float()
    t_len = x.shape[1]

    t_pred_full = None
    if teachers and kd_alpha > 0:
        with torch.no_grad():
            acc = None
            for t in teachers:
                tp, _ = t(x)
                acc = tp if acc is None else acc + tp
            t_pred_full = acc / len(teachers)

    seg = args.seg_len if args.seg_len > 0 else t_len
    first = 0 if args.loss_from_step == 0 else max(args.loss_from_step - (args.loss_from_step % seg), 0)
    # ensure the first segment covers indices >= loss_from_step
    while first + seg < args.loss_from_step and first + seg < t_len:
        first += seg
    opt.zero_grad(set_to_none=True)
    total, n_steps = 0.0, 0
    rho_total, rho_n = 0.0, 0
    rank_total, rank_n = 0.0, 0
    mse_total = 0.0
    h = None
    hg = None
    # Full-horizon correlation: the metric correlates over the WHOLE 20k
    # sequence, but with seg-len truncation each step only sees seg_len rows.
    # Targets are near-constant within a short window (autocorr 0.98 @lag1), so
    # a per-segment rho is dominated by noise and stays near zero. When
    # --corr-horizon seq, we accumulate the (already computed) predictions of
    # the earlier segments — detached, so memory stays at truncated-BPTT level —
    # and correlate over the whole horizon seen so far. Gradient flows through
    # the current segment only.
    acc_pred = [] if (args.corr_loss and args.corr_horizon in ("seq", "mid", "global")) else None
    acc_y = [] if acc_pred is not None else None
    _reset_every = int(getattr(args, "state_reset", 0) or 0)
    for start, end in _seg_bounds(t_len, first, seg, _reset_every):
        # --state-reset N: zero the hidden state at each block boundary that is a
        # multiple of N, instead of carrying it. The dozken reference measures
        # zero-state beating accumulated state after ~10k rows (reset2k/reset4k
        # global 0.6864/0.6823). Both directions of the carry (h and the
        # high-frequency modulator hg) are zeroed together -- a stale hg would
        # reintroduce the very history the reset removes.
        if _reset_every > 0 and start > 0 and start % _reset_every == 0:
            h = None
            hg = None
        # The reset must also CUT the segment when the reset grid is finer than
        # seg-len: with seg=8000 and reset=2000 every segment boundary (0/8000/
        # 16000) is already a multiple of 2000, so a reset only at segment
        # starts makes reset=2000, 4000 and 8000 bit-identical -- the arm is
        # never exercised, which is exactly what happened on the first run
        # (r2k and r4k scored bit-identical). The inner loop below splits every
        # seg-sized block at the reset boundaries inside it.
        x_seg = x[:, start:end, :]
        _twin_cell = getattr(model, "cell", "gru") in ("twinenc", "twin")
        if not _twin_cell:
            if getattr(model, "persist", None) is not None:   # D2: append the persistence col
                x_seg = model.persist(x_seg)
            # F2: the segment loop feeds model.gru DIRECTLY (bypasses forward), so the
            # mix must be applied here — same position as StatefulGRU.forward.
            if getattr(model, "feat_mix", None) is not None:
                x_seg = model.feat_mix(x_seg)
            if getattr(model, "feat_drop", ""):         # D-05: те же группы, что в forward
                x_seg = x_seg * model.feat_mask
            if getattr(model, "lag_attn", 0.0) > 0:      # P4: внимание по оси лагов
                x_seg = model.lag_gate(x_seg)
            if getattr(model, "hf_features", False):
                x_seg, hg = model.hf_modulate(x_seg, hg)
                if hg is not None:
                    hg = hg.detach()
        y_seg = y[:, start:end, :]
        if _twin_cell:
            # Twin-encoder: состояние — тройка (h0, h1, h); каждый элемент detach.
            if h is None:
                pred, h = model.twin(x_seg)
            else:
                pred, h = model.twin(x_seg, h)
            h = (h[0].detach(), h[1].detach(), h[2].detach())
            h_out = h[2]
        elif getattr(model, "cell", "gru") == "slstm":
            # sLSTM: состояние — 4-кортеж (h, c, m, n), каждый (layers, B, hid).
            if h is None:
                out, hn = model.slstm(x_seg)
            else:
                out, hn = model.slstm(x_seg, h)
            h = tuple(t.detach() for t in hn)
            h_out = out
            if getattr(model, "head_raw", False):
                out = torch.cat([out, x_seg], dim=-1)
            pred = model.head(model.slow_join(out))
        elif getattr(model, "cell", "gru") == "lstm":
            # LSTM: состояние — кортеж (h, c); out = выходы последнего слоя (B,T,hid),
            # то есть h_out трактуется так же, как у GRU.
            if h is None:
                out, (hn, cn) = model.rnn(x_seg)
            else:
                out, (hn, cn) = model.rnn(x_seg, h)
            h = (hn.detach(), cn.detach())
            h_out = out
            if getattr(model, "head_raw", False):     # D1: head sees [h, tick]
                out = torch.cat([out, x_seg], dim=-1)
            pred = model.head(model.slow_join(out))
        else:
            if getattr(model, "front", None) is not None:
                _p = model.front(x_seg)
                out, h = model.gru(_p, h) if h is not None else model.gru(_p)
                h = h.detach()
                h_out = out
                pred = model.head_d(model.post_ln(torch.cat([out, _p], dim=-1)))
            else:
                out, h = model.gru(x_seg, h) if h is not None else model.gru(x_seg)
                h = h.detach()
                h_out = out
                if getattr(model, "head_raw", False):     # D1: head sees [h, tick]
                    out = torch.cat([out, x_seg], dim=-1)
                pred = model.head(model.slow_join(out))
        # Twin-branch вычисляет prediction сам; ниже по u_basis/k_head не применяем.
        if not _twin_cell:
            # TZ-6 u-basis: the CORR term runs in (u+,u-), the MSE anchor stays in (t0,t1).
            # u-space is itself the "isolated denominators" construction — block_wp_torch
            # normalises each column by its OWN weighted variance, so the u- head stops
            # eclipsing u+ (c3: var_eff(u+)/var_eff(u-) = su_eff = 0.0399).
            pred_l, y_l = pred, y_seg
            if getattr(model, "u_basis", False):
                pred_l = model.u_split(pred, h_out)    # [g*u+, u-] as shipped
                pred = model.u_compose(pred, h_out)    # (t0,t1) for the MSE anchor
                y_l = torch.cat([(y_seg[..., 0:1] + y_seg[..., 1:2]) * INV_SQRT2,
                                 (y_seg[..., 0:1] - y_seg[..., 1:2]) * INV_SQRT2], dim=-1)
        else:
            pred_l, y_l = pred, y_seg
        # k-head и прочие downstream-преобразования — для twin не применяем (архитектура уже выдаёт 2 скаляра).
        _kce = None
        if getattr(model, "k_head_flag", False):  # 26.10: discrete-target head
            pred, _kce = model.k_apply(pred, h_out, y_seg)
        if tgt_ms is not None:                    # C1': та же нормировка, что у таргета
            pred = (pred - tgt_ms) / tgt_sd
        lo = max(args.loss_from_step - start, 0)
        if lo >= y_seg.shape[1]:
            continue
        # METRIC-ALIGNED CLIP (05.11): метрика считает и цель, и вес по clip(y,+-2),
        # а MSE-член брал сырой y. В данных |y| достигает 8.93, и 2.7% строк выходят за
        # клип: модель обучалась догонять выбросы, которые метрика обрезает. Корр-член
        # уже клипует (block_wp_torch), MSE — нет. --loss-clip 2.0 выравнивает постановку.
        _lc = float(getattr(args, "loss_clip", 0.0) or 0.0)
        yt = y_seg[:, lo:, :].clamp(-_lc, _lc) if _lc > 0 else y_seg[:, lo:, :]
        if _lc > 0:
            _n = int((yt != y_seg[:, lo:, :]).sum())
            _st = globals().setdefault("_LC_STAT", {"n": 0, "mx": 0.0, "batches": 0})
            _st["n"] += _n
            _st["batches"] += 1
            _st["mx"] = max(_st["mx"], float(y_seg.abs().max()))
        diff = pred[:, lo:, :] - yt
        if args.weighted_loss:
            if (tgt_delta is not None
                    and getattr(args, 'target_delta_raw_weights', False)):
                w = y_raw_abs[:, start + lo:end, :]   # plan v3: weights |clip(y,2)|
            else:
                w = yt.abs()
            if row_w is not None:
                w = w * row_w[:, start + lo:end].unsqueeze(-1)
                w = w + 1e-8        # keep the denominator finite when all masked
            if args.target_balance:
                # The metric averages t0 and t1 EQUALLY, but the weighted loss does
                # not: measured on 12 sequences, E|clip(t1,2)| = 0.688 versus 0.843
                # for t0, and 28.8% of t1 rows carry no weight at all versus 21.4%
                # for t0. So t1 receives ~18% less gradient while being scored the
                # same. Renormalise each target's contribution so both get equal
                # share of the MSE term.
                num = (w * diff * diff).sum(dim=(0, 1))              # (2,)
                den = w.sum(dim=(0, 1)).clamp_min(1e-6)              # (2,)
                mse_loss = (num / den).mean()                        # equal share
            else:
                mse_loss = (w * diff * diff).sum() / w.sum().clamp_min(1e-6)
        else:
            mse_loss = (diff ** 2).mean()
        # DELTA-LOSS (10.11.2026): цель почти всегда стоит на месте (90% удержаний), а линейная
        # регрессия на приращениях объясняет 56% дисперсии Δt0 (docs/COLUMN_FORENSICS_20261110.md).
        # Поэтому добавляем член на ПРИРАЩЕНИЯ предсказания: модель по-прежнему выдаёт уровень
        # (никакого интегрирования на инференсе, дрейф невозможен), но учится и динамике.
        _dw = float(getattr(args, "delta_loss", 0.0) or 0.0)
        if _dw > 0 and pred.shape[1] > 1:
            p_lo = pred[:, lo:, :]                       # same slice as yt (loss_from_step)
            dp = p_lo[:, 1:, :] - p_lo[:, :-1, :]
            dy = yt[:, 1:, :] - yt[:, :-1, :]
            wd = w[:, 1:, :]
            _dl = (wd * (dp - dy) ** 2).sum() / wd.sum().clamp_min(1e-6)
            mse_loss = mse_loss + _dw * _dl
        loss = mse_loss
        if _kce is not None:                      # 26.10: aux CE also without corr loss
            loss = loss + getattr(args, "k_head_ce", 0.3) * _kce
        rho_val = None
        # 28.10: rank_val/shock_val используются ниже вне блока `if args.corr_loss:`
        # (раньше они инициализировались только внутри него) => UnboundLocalError
        # при --corr-loss без --rank-mix. Инициализация здесь не меняет поведение
        # при corr_loss (там значения перезаписываются), но убирает падение.
        rank_val = None
        shock_val = None
        if args.corr_loss:
            if acc_pred is None:
                # per-segment horizon (cheap, but noisy -- see note above)
                rw_seg = None if row_w is None else row_w[:, start + lo:end]
                _ppc = int(getattr(args, "corr_pool_chunk", 0) or 0)
                if _ppc > 0:
                    rho = pooled_wp_chunked_torch(pred_l[:, lo:, :], y_l[:, lo:, :],
                                                  chunk=_ppc, row_w=rw_seg)
                else:
                    rho = block_wp_torch(pred_l[:, lo:, :], y_l[:, lo:, :], row_w=rw_seg)
                # ALROWS OBJECTIVE (plan FINAL_7SLOTS, A1/A2): mix in the unweighted
                # correlation over ALL rows -- the column the platform orders by.
                _umix1 = float(getattr(args, "corr_unit_mix", 0.0) or 0.0)
                if _umix1 > 0.0:
                    if _ppc > 0:
                        _ru = pooled_wp_chunked_torch(pred_l[:, lo:, :], y_l[:, lo:, :],
                                                      chunk=_ppc, row_w=None, unit=True)
                    else:
                        _ru = block_wp_torch(pred_l[:, lo:, :], y_l[:, lo:, :], unit=True)
                    rho = (1.0 - _umix1) * rho + _umix1 * _ru
            else:
                acc_pred.append(pred_l)
                acc_y.append(y_l)
                P = torch.cat([p if i == len(acc_pred) - 1 else p.detach()
                               for i, p in enumerate(acc_pred)], dim=1)
                Y = torch.cat(acc_y, dim=1)
                # S4 (mid horizon): cap the accumulated window at the last
                # --corr-mid-window steps instead of the whole sequence seen so far.
                # 'seq' correlates over everything (long, low-variance gradient),
                # 'seg' only within one segment (short, noisy). 'mid' is between.
                # The crop is applied to the tail, so the window START becomes
                # first + (Plen - keep); off is recomputed against that start.
                win_start = first
                if args.corr_horizon == "mid" and getattr(args, "corr_mid_window", 0):
                    keep = int(args.corr_mid_window)
                    if P.shape[1] > keep:
                        win_start = first + (P.shape[1] - keep)
                        P = P[:, -keep:, :]
                        Y = Y[:, -keep:, :]
                # restrict to steps >= loss_from_step (metric: required rows only)
                # NOTE: index the tail crop from win_start, NOT from `first` —
                # after cropping, P covers [win_start, win_start + P.shape[1]).
                off = max(0, args.loss_from_step - win_start)
                rw_seq = None if row_w is None else \
                    row_w[:, win_start + off:win_start + P.shape[1]]
                _chunk = int(getattr(args, "corr_chunk", 0) or 0)
                _ppc = int(getattr(args, "corr_pool_chunk", 0) or 0)
                rho = corr_term_torch(P[:, off:, :], Y[:, off:, :],
                                      horizon=args.corr_horizon,
                                      pool_chunk=_ppc, chunk=_chunk, row_w=rw_seq)
                # ALLROWS OBJECTIVE (plan FINAL_7SLOTS, A1/A2): mix in the unweighted
                # correlation over ALL rows (row_w=None, unit weights). 0.0 == old loss.
                _umix = float(getattr(args, "corr_unit_mix", 0.0) or 0.0)
                if _umix > 0.0:
                    _ru = corr_term_torch(P[:, off:, :], Y[:, off:, :],
                                          horizon=args.corr_horizon,
                                          pool_chunk=_ppc, chunk=_chunk, row_w=None, unit=True)
                    rho = (1.0 - _umix) * rho + _umix * _ru
                # DNPL: reward rho by the dispersion ratio, so shrinking the
                # output is no longer a free way to raise the correlation term.
                if getattr(args, "dnpl", 0.0) > 0.0:
                    rho_d = block_wp_dnpl_torch(P[:, off:, :], Y[:, off:, :], row_w=rw_seq)
                    _d = float(args.dnpl)
                    rho = (1.0 - _d) * rho + _d * rho_d
                # LEVEL-AWARE term: the per-sequence LEVEL of the prediction,
                # which block_wp_torch is exactly blind to (see level_corr_torch).
                _lmix = getattr(args, "level_mix", 0.0)
                if _lmix > 0.0:
                    rho_lvl = level_corr_torch(P[:, off:, :], Y[:, off:, :], row_w=rw_seq)
                    rho = (1.0 - _lmix) * rho + _lmix * rho_lvl
                # SCALE term: penalise output SHRINKAGE per sequence.
                # The measured defect (21.10 14:55): in the hard class the model
                # halves the spread of its output (std(pred) 0.5828 vs
                # std(y) 0.8165) => hard_r 0.4714 instead of 0.73-0.77.
                # block_wp is blind to it (correlation is scale-invariant) and a
                # post-hoc rescale does nothing (Pearson). The oracle fix is
                # worth +0.058 hard_r, so this is the only lever that reaches
                # the heavy group directly.
                # ATTENUATION term (21.10 16:45): the measured prize. The oracle
                # alpha_s := 1 gains +0.0143 pooled / +0.0342 hard_r / +0.0109
                # s60p on the champion's dump -- s60p is the slot gate, and until
                # now no axis moved it. The penalty is scale-ANCHORED (see
                # src/atten_loss.py): relu(alpha*std(y) - std(p))/std(y), so it
                # GROWS when the output is squeezed. The first implementation,
                # relu(1-alpha), is degenerate: alpha is inverse-homogeneous
                # (alpha(k*p) = alpha(p)/k), so squeezing the output drove that
                # penalty to ZERO -- it rewarded the very defect we are fixing.
                # POOLED term: the EXACT platform metric over the whole batch.
                # Removes the level-vs-shape trade-off by construction.
                _pmix = getattr(args, "pooled_mix", 0.0)
                if _pmix > 0.0:
                    rho_pool = pooled_wp_torch(P[:, off:, :], Y[:, off:, :], row_w=rw_seq)
                    rho = (1.0 - _pmix) * rho + _pmix * rho_pool
            rho_val = float(rho.detach())
            loss = args.corr_mix * (1.0 - rho) + (1.0 - args.corr_mix) * mse_loss
            if _kce is not None:                  # 26.10: aux CE over the integer tick count
                loss = loss + getattr(args, "k_head_ce", 0.3) * _kce
            # DRIFT (12.11): on rows where the target does NOT move, penalise movement of the
            # prediction. Measured: WP is 0.744 right after a level change but 0.668 at age>=20
            # steps, which carries 32% of the metric weight -- the model holds the level badly.
            # Scaled by mean|y| so the term is unit-free.
            _dmix = getattr(args, "drift_mix", 0.0)
            if _dmix > 0.0:
                _dpv = P[:, off + 1:, :] - P[:, off:-1, :]
                _dyv = Y[:, off + 1:, :] - Y[:, off:-1, :]
                _flat = (_dyv.abs() < 1e-9).float()
                _pmd = (_dpv.abs() * _flat).sum() / (_flat.sum() + 1.0)
                loss = loss + _dmix * _pmd / (Y[:, off:, :].abs().mean() + 1e-6)
            # --- the penalty terms, AFTER the loss is assembled (the bug fix) ---
            # SCALE: penalise the output SHRINKAGE per sequence.
            _smix = getattr(args, "scale_mix", 0.0)
            if _smix > 0.0:
                _pen = scale_shrink_torch(P[:, off:, :], Y[:, off:, :], row_w=rw_seq)
                loss = loss + _smix * _pen
            # ATTENUATION: the scale-anchored penalty from src/atten_loss.py.
            _amix = getattr(args, "atten_mix", 0.0)
            if _amix > 0.0:
                _pen = atten_shrink_torch(P[:, off:, :], Y[:, off:, :], row_w=rw_seq)
                loss = loss + _amix * _pen
                _ad = atten_diag_torch(P[:, off:, :], Y[:, off:, :], row_w=rw_seq)
                for _k, _v in _ad.items():
                    atten_diag_acc[_k] = atten_diag_acc.get(_k, 0.0) + _v
                atten_diag_n[0] += 1
            # H2: add a rank (Spearman-shaped) term alongside the Pearson term.
            # Framed strictly as rank vs value, one variable: the Pearson term is
            # untouched and the rank term enters with its own dose.
            if getattr(args, "rank_mix", 0.0) > 0:
                if acc_pred is None:
                    rw_rank = None if row_w is None else row_w[:, start + lo:end]
                    rho_s = block_spearman_torch(pred[:, lo:, :], y_seg[:, lo:, :],
                                                 row_w=rw_rank,
                                                 tau=getattr(args, "rank_tau", 0.05))
                else:
                    rw_rank = None if row_w is None else \
                        row_w[:, win_start + off:win_start + P.shape[1]]
                    rho_s = block_spearman_torch(P[:, off:, :], Y[:, off:, :],
                                                 row_w=rw_rank,
                                                 tau=getattr(args, "rank_tau", 0.05))
                loss = loss + args.rank_mix * (1.0 - rho_s)
                rank_val = float(rho_s.detach())
            else:
                rank_val = None
            # A2 SHOCK term (24.10): innovations' correlation, one field.
            _shock = getattr(args, "shock_mix", 0.0)
            if _shock > 0.0:
                if acc_pred is None:
                    rw_sh = None if row_w is None else row_w[:, start + lo:end]
                    _sh = shock_wp_torch(pred[:, lo:, :], y_seg[:, lo:, :], row_w=rw_sh)
                else:
                    rw_sh = None if row_w is None else \
                        row_w[:, win_start + off:win_start + P.shape[1]]
                    _sh = shock_wp_torch(P[:, off:, :], Y[:, off:, :], row_w=rw_sh)
                loss = loss - _shock * _sh
                shock_val = float(_sh.detach())
            else:
                shock_val = None
        if t_pred_full is not None:
            diff_t = pred[:, lo:, :] - t_pred_full[:, start + lo:end, :]
            loss = loss + kd_alpha * (diff_t * diff_t).mean()
        # P1 (исправлено 05.11): слагаемое самопредсказания добавляется ПОСЛЕ сборки лосса.
        # Раньше оно стояло до corr-ветки, а та пересобирает loss целиком и затирала его →
        # прогоны с aux и без давали БИТ-ИДЕНТИЧНЫЕ веса (проверено, F354).
        if getattr(model, "aux_head", None) is not None:
            _lag = max(1, int(getattr(model, "aux_lag", 1)))
            if h_out.shape[1] > _lag:
                _ap = model.aux_head(h_out[:, :-_lag, :])
                _ay = x_seg[:, _lag:, :]
                loss = loss + float(getattr(model, "aux_mix", 0.0)) * ((_ap - _ay) ** 2).mean()
        (loss / max(1, (t_len - first + seg - 1) // seg)).backward()
        total += float(loss.detach())
        if rho_val is not None:
            rho_total += rho_val
            rho_n += 1
        if rank_val is not None:
            rank_total += rank_val
            rank_n += 1
        mse_total += float(mse_loss.detach())
        n_steps += 1
    if args.grad_clip:
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    opt.step()
    info = {}
    if rho_n:
        info["rho"] = rho_total / rho_n
    if rank_n:
        info["rank"] = rank_total / rank_n
    info["mse"] = mse_total / max(n_steps, 1)
    if atten_diag_n[0]:
        info["atten"] = {k: v / atten_diag_n[0] for k, v in atten_diag_acc.items()}
    return total / max(n_steps, 1), x.shape[0] * t_len, info


def _install_hard_kappa(args, pf, n_groups):
    """A1 hard-carrier dose (24.10): build the per-row-group hard-kappa map.

    Multiplies the EXACT-mask row weights of hard-carrier groups (class 3,
    delta 0.65687) by --row-weight-hard-kappa on the VALID side only. One
    field vs the control cell; inert at 1.0 (loss bit-identical).
    """
    _hk = getattr(args, "row_weight_hard_kappa", 1.0)
    if _hk == 1.0:
        return
    if args.row_weight_hybrid is None:
        raise SystemExit("--row-weight-hard-kappa requires --row-weight-hybrid")
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                      "..", "tools_class"))
    import classmap as _cm
    cm = _cm.load()
    assert cm.hard_idx == 3, f"hard_idx {cm.hard_idx} != 3 — labels moved?"
    n_valid = n_groups - args.row_weight_hybrid
    assert n_valid == len(cm.class_seq), \
        f"valid groups {n_valid} != classmap {len(cm.class_seq)}"
    globals()["_HARD_KAPPA"] = {i: float(_hk)
                                for i, c in enumerate(cm.class_seq) if c == 3}
    print(f"[train] A1 hard-kappa ON: {len(globals()['_HARD_KAPPA'])} hard groups "
          f"x{_hk}", flush=True)


def _get_pf(path):
    """Thread-local ParquetFile cache. Opening the file costs ~1.8 s over 9p
    (footer with 10k+ row groups), so each reader thread opens it once."""
    global _PF_LOCK
    import threading
    if _PF_LOCK is None:
        _PF_LOCK = threading.Lock()
    key = (threading.get_ident(), path)
    with _PF_LOCK:
        pf = _PF_CACHE.get(key)
    if pf is None:
        # minimal: retry transient ENOMEM — drvfs/9p reads can fail with
        # errno 12 while the VM is under memory pressure (see notes).
        for attempt in range(4):
            try:
                pf = pq.ParquetFile(path, memory_map=False)
                break
            except (OSError, MemoryError):
                if attempt == 3:
                    raise
                time.sleep(2 + 3 * attempt)
        with _PF_LOCK:
            _PF_CACHE[key] = pf
    return pf


def read_batch(parquet_path, group_ids, columns, use_threads=False, feat_mode="raw",
               weight_col=None, hybrid_split=None, mask_path=None, exact_kappa=1.0,
               train_ones=False, mask_p=None):
    """Read several row groups into (B,20000,F) features + (B,20000,2) targets.

    feat_mode: "raw" = the 112 columns as-is; "eng" = engineered features from
    feat_eng.make_features (see that module for the verified quote columns and
    the docs warning that the index is NOT a depth level).
    """
    pf = _get_pf(parquet_path)
    fd = _get_fd(parquet_path)
    feats, targs, wrows = [], [], []
    read_cols = list(columns)
    if weight_col is not None and weight_col not in read_cols and hybrid_split is None:
        read_cols.append(weight_col)
    for g in group_ids:
        # minimal: retry transient ENOMEM on drvfs/9p reads (kernel read under
        # memory pressure); behaviour unchanged when reads succeed.
        for attempt in range(4):
            try:
                table = pf.read_row_group(g, columns=read_cols, use_threads=use_threads)
                break
            except (OSError, MemoryError):
                if attempt == 3:
                    raise
                time.sleep(2 + 3 * attempt)
        drop_cache_range(pf, fd, g)
        f = np.column_stack([table[c].to_numpy(zero_copy_only=False) for c in FEATURE_COLUMNS])
        t = np.column_stack([table[c].to_numpy(zero_copy_only=False) for c in TARGET_COLUMNS])
        f = f.astype(np.float32, copy=False)
        if feat_mode == "eng":
            from feat_eng import make_features
            f = make_features(f)
        elif feat_mode == "stream336":
            from stream336 import stream336_np
            f = stream336_np(f)
        elif feat_mode == "delta224":
            from delta224 import delta224_np
            f = delta224_np(f)
        elif feat_mode == "win":
            from winfeat import win_np
            f = win_np(f)
        feats.append(f)
        targs.append(t.astype(np.float32, copy=False))
        if weight_col is not None:
            if hybrid_split is not None:
                if g < hybrid_split:
                    if train_ones:
                        # factorial cell: uniform train-side weights
                        wrows.append(np.ones(f.shape[0], dtype=np.float32))
                    else:
                        # train-side group: no mask column anywhere -> p_hat proxy.
                        # patch_mask112: honour the coeffs' width (112 = full raw row,
                        # 8 = legacy a-slice). Default behaviour unchanged.
                        if getattr(_PHAT_W_NP, "shape", (0,))[0] == 112:
                            z = f @ _PHAT_W_NP + _PHAT_B_NP
                        else:
                            a8 = f[:, 104:112]
                            z = a8 @ _PHAT_W_NP + _PHAT_B_NP
                        ph = 1.0 / (1.0 + np.exp(-z))
                        wrows.append(ph.astype(np.float32))
                else:
                    # valid-side group: EXACT mask from the aligned mask file
                    mpf = _get_pf(mask_path)
                    mt = mpf.read_row_group(g - hybrid_split, columns=[weight_col])
                    _we = mt[weight_col].to_numpy(zero_copy_only=False).astype(np.float32)
                    if exact_kappa != 1.0:
                        _we = _we * np.float32(exact_kappa)
                    # C2' sampling-robust (24.10): bernoulli(mask_p) thinning of the
                    # EXACT side, a FRESH draw per row-group read — the mask becomes
                    # a random subset each epoch (robust supervised training across
                    # diverse sampling variations, PRISM). inert at mask_p None/1.0.
                    if mask_p is not None and mask_p < 1.0:
                        _d = globals().get("_MASK_RNG")
                        _r = np.random.default_rng(_d) if _d is not None else \
                            np.random.default_rng()
                        _keep = (_r.random(_we.shape[0]) < mask_p)
                        _we = _we * _keep.astype(np.float32)
                    # A1 hard-carrier dose: multiply rated rows of the HARD class
                    # (delta 0.65687) by --row-weight-hard-kappa. The carrier class
                    # comes from the classmap (artifact, and is INERT on the train
                    # side by construction). One-field A/B vs the same cell without it.
                    _hk = globals().get("_HARD_KAPPA", {}).get(g - hybrid_split)
                    if _hk is not None:
                        _we = _we * np.float32(_hk)
                    wrows.append(_we)
            else:
                wrows.append(table[weight_col].to_numpy(zero_copy_only=False).astype(np.float32))
    if weight_col is not None:
        return np.stack(feats), np.stack(targs), np.stack(wrows)
    return np.stack(feats), np.stack(targs)


def split_windows(feats, targs, wrow, win):
    """S-b: cut each T-row sequence into consecutive windows of `win` rows.

    WHY: every training example currently carries the full 20000-row sequence
    with truncated BPTT over seg_len. Cutting each 20k sequence into 10k halves
    doubles the number of INDEPENDENT gradient trajectories (zero hidden at the
    start of each half) without adding a single row of data. RESEARCH_02 §5
    describes this as "separate examples with zero hidden" — a different regime
    from the current seg-len truncation, which carries the hidden ACROSS segments
    within one sequence.

    The batch dimension grows by the number of windows per sequence, so the
    effective batch becomes B * (T/win). Callers must shrink --batch accordingly
    to keep the memory footprint constant.

    Returns (feats, targs, wrow) with the window dimension folded into batch.
    """
    if win <= 0 or feats.shape[1] <= win:
        return feats, targs, wrow
    t_len = feats.shape[1]
    n_win = t_len // win
    used = n_win * win                       # drop the tail remainder
    f = feats[:, :used, :].reshape(-1, win, feats.shape[2])
    t = targs[:, :used, :].reshape(-1, win, targs.shape[2])
    w = None
    if wrow is not None:
        w = wrow[:, :used].reshape(-1, win)
    return f, t, w


_GROUP_KEEP_CACHE = {}


def load_group_list(path):
    """Group ids to train on (regime specialists). Newline list or JSON array. Cached."""
    if path in _GROUP_KEEP_CACHE:
        return _GROUP_KEEP_CACHE[path]
    text = open(path).read().strip()
    if text.startswith("["):
        ids = {int(x) for x in json.loads(text)}
    else:
        ids = {int(line) for line in text.splitlines() if line.strip()}
    _GROUP_KEEP_CACHE[path] = ids
    return ids


def epoch_group_order(n_groups, max_seq, rng, exact_split=None, oversample=1, start=0,
                      group_stride=0, group_offset=0, group_keep=None):
    """Row-group ids for one epoch, read SEQUENTIALLY from the file.

    Random access through the 9p bridge (/mnt/d) is ~10x slower than
    sequential reads, and the dataset is pre-shuffled by the organizers
    (seq_ix is random), so we sweep the file in order starting at a random
    offset each epoch instead of shuffling group ids.

    exact_split/oversample (mask branch step 2b): when exact_split is set and
    oversample > 1, every group id >= exact_split is appended (oversample - 1)
    extra times — the "dose" of the exact-mask side goes up oversample-fold.

    group_keep (10.11): when set, only these group ids are used — regime
    specialists train on one volatility regime at a time.
    """
    n = n_groups if max_seq is None else min(n_groups, max_seq)
    ids = list(range(start, n))
    if group_keep is not None:
        ids = [i for i in ids if i in group_keep]
    # BAGGING (09.11): keep only ids with (i-start) % stride == offset, so several
    # models can be trained on disjoint sequence subsets (decorrelated errors).
    if group_stride and group_stride > 1:
        ids = [i for i in ids if (i - start) % group_stride == (group_offset % group_stride)]
    if max_seq is None:
        off = int(rng.integers(len(ids)))
        ids = ids[off:] + ids[:off]
    if exact_split is not None and oversample > 1:
        tail = [i for i in ids if i >= exact_split]
        ids = ids + tail * (oversample - 1)
    return ids


def batches_in_order(ids, batch_size, rng):
    """Yield batches in file order; shuffle within each batch (free)."""
    for i in range(0, len(ids), batch_size):
        b = ids[i:i + batch_size]
        rng.shuffle(b)
        yield b


def prefetch(iterable, fn, workers=3):
    """Yield fn(item) with a bounded thread pool reading ahead."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = deque()
        for item in iterable:
            futures.append(ex.submit(fn, item))
            if len(futures) >= workers:
                yield futures.popleft().result()
        while futures:
            yield futures.popleft().result()


def export_onnx_twin(model, out_path, hid, mean=None, std=None):
    """Пошаговый ONNX-экспорт для twin-encoder (контракт 3-in/3-out, H-twin)."""
    from twin_arch import TwinEncGRU  # noqa: F401
    mode = getattr(model, "cell", "twinenc")
    if mode not in ("twinenc", "twin"):
        mode = "twinenc"
    wrapper = StatefulTwin(width=hid, mode=mode)
    if hasattr(model, "twin") and model.twin is not None:
        wrapper.twin.load_state_dict(model.twin.state_dict())
    if mean is not None and std is not None:
        wrapper.set_norm(mean, std)
    wrapper.eval()
    w = int(hid)
    feats = torch.zeros(1, 1, N_FEATURES, device="cpu")
    h0 = torch.zeros(1, 1, 2 * w, device="cpu")
    h1 = torch.zeros(1, 1, w // 2, device="cpu")
    torch.onnx.export(
        wrapper, (feats, h0, h1), str(out_path),
        input_names=["features", "hidden_0", "hidden_1"],
        output_names=["prediction", "next_hidden_0", "next_hidden_1"],
        opset_version=17, do_constant_folding=True,
    )
    return


def export_onnx_slstm(model, out_path, hid, layers, mean=None, std=None):
    """Пошаговый ONNX sLSTM: 5 входов / 5 выходов.

    Обобщённый список состояний: hidden_0..hidden_3 = h, c, m, n, каждый (layers,1,hid).
    solution_generic.py обходит ВСЕ state-входы по именам и берёт successor как
    "next_"+name, поэтому 4 состояния не требуют правок в сабмишене.
    """
    wrapper = StatefulSLSTM(hid=int(hid), layers=int(layers), n_features=int(model.n_features))
    wrapper.slstm.load_state_dict(model.slstm.state_dict())
    wrapper.head.load_state_dict(model.head.state_dict())
    if mean is not None and std is not None:
        wrapper.set_norm(mean, std)
    wrapper.eval()
    L = int(layers)
    feats = torch.zeros(1, 1, int(model.n_features))
    states = [torch.zeros(L, 1, int(hid)) for _ in range(4)]
    names = [f"hidden_{i}" for i in range(4)]
    torch.onnx.export(
        wrapper, (feats, *states), str(out_path),
        input_names=["features"] + names,
        output_names=["prediction"] + ["next_" + n for n in names],
        # opset 18: chunk() lowers to Split(num_outputs=...), which the 18->17
        # version converter cannot express (ORT: "Unrecognized attribute: num_outputs").
        opset_version=18, do_constant_folding=True,
    )


def export_onnx(model, out_path, hid, layers):
    if getattr(model, "slow_mix", 0.0) > 0 or getattr(model, "lag_attn", 0.0) > 0:      # P2: экспорт требует поддержки в C-ядре
        print("[train] ONNX-экспорт пропущен: медленное состояние (P2) не реализовано в ядре.",
              flush=True)
        return
    _cell = getattr(model, "cell", "gru")
    if _cell == "slstm":
        export_onnx_slstm(model, out_path, hid, layers)
        return
    if _cell == "lstm":
        # 10.11.2026: раньше здесь был молчаливый `return` - тренер печатал "DONE ... onnx=<path>",
        # файла не было, и очередь падала на "FATAL: onnx missing" уже после прогона (div_lstm).
        # Честнее упасть сразу и сказать, чем делать вид, что экспорт состоялся.
        raise SystemExit(
            "[train] EXPORT: LSTM не экспортируется через StatefulGRU-обёртку (нужен пошаговый "
            "ONNX с состояниями h,c). Чекпойнт сохранён - считай его напрямую: "
            "tools_class/score_ckpt_direct.py --ckpt ...pt --save-npz ...npz"
        )
    if _cell in ("twinenc", "twin"):
        export_onnx_twin(model, out_path, hid)
        return
    use_eng = getattr(model, "use_eng", False)
    nf = N_FEATURES if use_eng else int(getattr(model, "n_features", N_FEATURES))
    head_raw = getattr(model, "head_raw", False)
    head_mlp = getattr(model, "head_mlp", False)
    persist_col = getattr(model, "persist_col", -1)
    if persist_col >= 0:
        nf -= 1   # StatefulGRU re-adds the persist column in its own __init__
    feat_gate = getattr(model, "feat_gate", False)
    feat_mix = getattr(model, "feat_mix", None) is not None
    u_basis = getattr(model, "u_basis", False)
    gamma_gate = getattr(model, "gamma", None) is not None
    k_head_on = bool(getattr(model, "k_head_flag", False))
    if k_head_on:
        raise SystemExit(
            "REFUSED: --k-head is not ONNX-exportable: the offline model pools tick_hat over the whole "
            "sequence while the streaming wrapper applies cls_head per step (predictions differ by ~1e-3). "
            "Retrain without --k-head before packaging.")
    wrapper = StatefulGRU(hid=hid, layers=layers, n_features=nf, use_eng=use_eng,
                          head_raw=head_raw, head_mlp=head_mlp, persist_col=persist_col, feat_gate=feat_gate,
                          feat_mix=feat_mix, u_basis=u_basis, gamma_gate=gamma_gate,
                          k_head=k_head_on, k_lambda=float(getattr(model, "k_lambda", 0.5)),
                          front_dim=int(getattr(model, "front_dim", 0)))
    if use_eng:
        wrapper.feat.load_state_dict(model.feat.state_dict())
    wrapper.gru.load_state_dict(model.gru.state_dict())
    wrapper.head.load_state_dict(model.head.state_dict())
    if getattr(model, "front_dim", 0):
        wrapper.front.load_state_dict(model.front.state_dict())
        wrapper.post_ln.load_state_dict(model.post_ln.state_dict())
        wrapper.head_d.load_state_dict(model.head_d.state_dict())
    if feat_gate:
        wrapper.gate.load_state_dict(model.gate.state_dict())
    if feat_mix:                                    # F2
        wrapper.feat_mix.load_state_dict(model.feat_mix.state_dict())
    if gamma_gate:                                  # TZ-6
        wrapper.gamma.load_state_dict(model.gamma.state_dict())
    if k_head_on:                                   # 26.10
        wrapper.k_head.load_state_dict(model.k_head.state_dict())
        wrapper.cls_head.load_state_dict(model.cls_head.state_dict())
    wrapper.eval()
    dev = "cpu"
    feats = torch.zeros(1, 1, nf, device=dev)
    h0 = torch.zeros(1, 1, hid, device=dev)
    h1 = torch.zeros(1, 1, hid, device=dev)
    torch.onnx.export(
        wrapper, (feats, h0, h1), str(out_path),
        input_names=["features", "hidden_0", "hidden_1"],
        output_names=["prediction", "next_hidden_0", "next_hidden_1"],
        opset_version=17, do_constant_folding=True,
    )


def export_onnx_norm(model, out_path, hid, layers, mean, std):
    """Export the stateful wrapper WITH baked-in input standardization."""
    if getattr(model, "slow_mix", 0.0) > 0 or getattr(model, "lag_attn", 0.0) > 0:      # P2: экспорт требует поддержки в C-ядре
        print("[train] ONNX-экспорт пропущен: медленное состояние (P2) пока не реализовано "
              "в ядре; обучение и локальная оценка работают.", flush=True)
        return
    _cell = getattr(model, "cell", "gru")
    if _cell == "slstm":
        export_onnx_slstm(model, out_path, hid, layers, mean=mean, std=std)
        return
    if _cell in ("twinenc", "twin"):
        export_onnx_twin(model, out_path, hid, mean=mean, std=std)
        return
    use_eng = getattr(model, "use_eng", False)
    nf = N_FEATURES if use_eng else int(getattr(model, "n_features", N_FEATURES))
    head_raw = getattr(model, "head_raw", False)
    head_mlp = getattr(model, "head_mlp", False)
    persist_col = getattr(model, "persist_col", -1)
    if persist_col >= 0:
        nf -= 1   # StatefulGRU re-adds the persist column in its own __init__
    feat_gate = getattr(model, "feat_gate", False)
    feat_mix = getattr(model, "feat_mix", None) is not None
    u_basis = getattr(model, "u_basis", False)
    gamma_gate = getattr(model, "gamma", None) is not None
    k_head_on = bool(getattr(model, "k_head_flag", False))
    if k_head_on:
        raise SystemExit(
            "REFUSED: --k-head is not ONNX-exportable: the offline model pools tick_hat over the whole "
            "sequence while the streaming wrapper applies cls_head per step (predictions differ by ~1e-3). "
            "Retrain without --k-head before packaging.")
    wrapper = StatefulGRU(hid=hid, layers=layers, n_features=nf, use_eng=use_eng,
                          head_raw=head_raw, head_mlp=head_mlp, persist_col=persist_col, feat_gate=feat_gate,
                          feat_mix=feat_mix, u_basis=u_basis, gamma_gate=gamma_gate,
                          k_head=k_head_on, k_lambda=float(getattr(model, "k_lambda", 0.5)),
                          front_dim=int(getattr(model, "front_dim", 0)))
    if use_eng:
        wrapper.feat.load_state_dict(model.feat.state_dict())
    wrapper.gru.load_state_dict(model.gru.state_dict())
    wrapper.head.load_state_dict(model.head.state_dict())
    if getattr(model, "front_dim", 0):
        wrapper.front.load_state_dict(model.front.state_dict())
        wrapper.post_ln.load_state_dict(model.post_ln.state_dict())
        wrapper.head_d.load_state_dict(model.head_d.state_dict())
    if feat_gate:
        wrapper.gate.load_state_dict(model.gate.state_dict())
    if feat_mix:                                    # F2
        wrapper.feat_mix.load_state_dict(model.feat_mix.state_dict())
    if gamma_gate:                                  # TZ-6
        wrapper.gamma.load_state_dict(model.gamma.state_dict())
    if k_head_on:                                   # 26.10
        wrapper.k_head.load_state_dict(model.k_head.state_dict())
        wrapper.cls_head.load_state_dict(model.cls_head.state_dict())
    wrapper.set_norm(mean, std)
    wrapper.eval()
    feats = torch.zeros(1, 1, nf)
    h0 = torch.zeros(1, 1, hid)
    h1 = torch.zeros(1, 1, hid)
    torch.onnx.export(
        wrapper, (feats, h0, h1), str(out_path),
        input_names=["features", "hidden_0", "hidden_1"],
        output_names=["prediction", "next_hidden_0", "next_hidden_1"],
        opset_version=17, do_constant_folding=True,
    )


def _proxymask_selftest(device="cpu"):
    """A2 proxy-mask selftest (0 GPU, 0 data): 5 synthetic batches check that
    (1) hard weights are exactly in {0,1}, (2) the selected fraction matches the
    expected one (threshold: measured vs analytic; topk: exactly K/T per sequence
    except ties), (3) weights agree with an independent numpy recomputation,
    (4) no NaN anywhere and the gradient reaches the loss through the weights."""
    import json as _json
    print("[selftest] proxy-mask selftest (5 synthetic batches, B=4, T=512)", flush=True)
    device = torch.device(device)
    coeff_path = _phat_coeffs_path("tmp/mask_logit_coeffs.json")
    with open(coeff_path) as fh:
        c = _json.load(fh)
    w_np = np.asarray(c["w_raw"], dtype=np.float64)
    b_np = float(c["b_raw"])
    print(f"[selftest] coeffs: {coeff_path} "
          f"(auc_holdout={c.get('auc_holdout', 'n/a')})", flush=True)

    global _PHAT_W, _PHAT_B, _PHAT_NF
    _PHAT_W = torch.tensor(w_np, dtype=torch.float32, device=device)
    _PHAT_B = torch.tensor(b_np, dtype=torch.float32, device=device)
    _PHAT_NF = len(w_np)

    g_cpu = torch.Generator().manual_seed(0)
    w_lin = torch.nn.Linear(N_FEATURES, 2).to(device)
    n_bad = 0
    for i in range(5):
        x_raw = torch.randn(4, 512, N_FEATURES, generator=g_cpu).to(device)
        a8 = x_raw[..., 104:112].double()
        p_ref = torch.sigmoid(a8 @ torch.from_numpy(w_np) + b_np)
        # (2) threshold mode: expected analytic fraction vs helper's
        thr = 0.33
        args = SimpleNamespace(phat_select="threshold", phat_threshold=thr,
                               phat_topk=None)
        row_w, frac = _phat_hard_weights(p_ref, "threshold", thr, None)
        ref_hard = (p_ref > thr).to(x_raw.dtype)
        ok_01 = bool(((row_w == 0.0) | (row_w == 1.0)).all())
        ok_match = bool(torch.equal(row_w, ref_hard))
        an_frac = float((a8 @ torch.from_numpy(w_np) + b_np >
                         math.log(thr / (1.0 - thr))).float().mean())
        # (2b) topk mode: exactly K ones per sequence when no ties
        K = 60  # 60/512 = 11.72% (inside the 8-12% mask dose)
        row_wk, frac_k = _phat_hard_weights(p_ref.double(), "topk", None, K)
        counts = row_wk.sum(dim=-1)
        ok_k = bool(torch.all(counts == K))
        # (4) no NaN + gradient reaches the loss through the weights
        pred = w_lin(x_raw)                                   # (B,T,2) fake model
        loss = (row_w.unsqueeze(-1) * pred * pred).sum() \
            / row_w.sum().clamp_min(1e-8)                     # weighted mean square
        loss.backward()
        ok_nan = bool(torch.isfinite(loss)) and bool(
            torch.isfinite(w_lin.weight.grad).all())
        grad_norm = float(w_lin.weight.grad.norm())
        print(f"[selftest] batch {i}: threshold t={thr} "
              f"frac={frac:.4f} analytic={an_frac:.4f} | "
              f"topk K={K} counts={counts.tolist()} frac={frac_k:.4f} | "
              f"in01={ok_01} match_ref={ok_match} exactK={ok_k} "
              f"loss={loss.item():.6f} grad_norm={grad_norm:.4f} "
              f"finite={ok_nan}", flush=True)
        n_bad += (not ok_01) + (not ok_match) + (not ok_k) + (not ok_nan) \
            + (abs(frac - an_frac) > 1e-6) + float(grad_norm < 1e-8)
        w_lin.zero_grad(set_to_none=True)
    print(f"[selftest] RESULT: {'PASS' if n_bad == 0 else 'FAIL'} "
          f"({n_bad} failed checks over 5 batches)", flush=True)
    return n_bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hid", type=int, default=96)
    ap.add_argument("--front-proj", type=int, default=0,
                    help="David Alami front-end: Linear(F->N)+GELU+LayerNorm, GRU over the "
                         "projection, LayerNorm(concat[gru_out, proj]) + MLP head. 0 = off.")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hf-features", action="store_true",
                    help="H-F: gen GRU-16 + per-group (4) scale/shift модуляция входа")
    ap.add_argument("--hf-causal-mod", action="store_true",
                    help="H-F: scale/shift каждого шага из его СОБСТВЕННОГО состояния "
                         "(каузально; без флага — legacy look-ahead из последнего шага сегмента)")
    ap.add_argument("--cell-type", choices=["gru", "lstm", "slstm", "twinenc", "twin"], default="gru",
                    help="ячейка рекуррентного слоя: gru (legacy), lstm, twinenc (8 кодировщиков) или twin")
    ap.add_argument("--state-reset", type=int, default=0,
                    help="zero the GRU hidden state every N rows instead of carrying it "
                         "(0 = carry, current behaviour). dozken: reset2k/4k beats carry.")
    ap.add_argument("--delta-loss", type=float, default=0.0,
                    help="10.11.2026: add W * weighted MSE(pred[t]-pred[t-1], y[t]-y[t-1]). "
                         "The output stays a LEVEL (no integration, no drift) but the model is "
                         "also trained on the change, which is linearly 56%% explainable "
                         "(docs/COLUMN_FORENSICS_20261110.md). 0.0 = off (byte-identical).")
    ap.add_argument("--seg-len", type=int, default=4000,
                    help="truncated-BPTT segment length; 0 = full 20k backprop")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-seq", type=int, default=None, help="limit sequences (smoke)")
    ap.add_argument("--seq-start", type=int, default=0,
                    help="first sequence id (absolute row-group index). Used for the "
                         "two-stage scheme: base on train, then fine-tune on the valid "
                         "range [seq-start, max-seq) that is closest to the test regime.")
    ap.add_argument("--loss-from-step", type=int, default=WARMUP,
                    help="first step included in loss (99=required rows only)")
    ap.add_argument("--prefetch-workers", type=int, default=6,
                    help="reader threads ahead of the GPU step; 3 was the old default, "
                         "6 keeps the 7x-faster GPU fed (see cudnn.benchmark note in main)")
    ap.add_argument("--reader-use-threads", action="store_true",
                    help="let pyarrow use internal threads inside reader workers")
    ap.add_argument("--tag", default="m1")
    ap.add_argument("--init", default=None, help="warm-start: load state_dict from this .pt")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--input-dropout", type=float, default=0.0,
                    help="drop each input feature with prob p during training (scaled by 1/(1-p))")
    ap.add_argument("--group-stride", type=int, default=0,
                    help="bagging: keep only groups with (i-seq_start) modulo stride == offset")
    ap.add_argument("--group-offset", type=int, default=0)
    ap.add_argument("--group-list-file", default=None,
                    help="regime specialist: train only on group ids from this file (newline or JSON)")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--lr-decay", choices=["none", "cosine", "linear", "step", "exp"],
                    default="none",
                    help="LR schedule across epochs. 'linear' is the non-cosine "
                         "annihilation (21.10, program S cell S3)")
    ap.add_argument("--min-lr", type=float, default=1e-5,
                    help="floor for cosine decay")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="dropout between GRU layers (train only, eval=0)")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--kd-ckpts", nargs="*", default=None,
                    help="knowledge distillation: teacher checkpoints (.pt)")
    ap.add_argument("--kd-alpha", type=float, default=0.0,
                    help="weight of KD loss (MSE to teachers' mean prediction)")
    ap.add_argument("--target-standardize", action="store_true",
                    help="C1' (MA2 20.10): loss on per-sequence standardized targets "
                         "y'=(y-m_s)/sigma_s (weighted by |y| over scored rows); prediction "
                         "normalized with the same (m_s, sigma_s) BEFORE the loss. One variable.")
    ap.add_argument("--target-delta", action="store_true",
                    help="B-A (plan v3 13.11): loss on the dimensionless ticks k=y/delta_s, "
                         "delta_s = min|y|>0 of the sequence; the model OUTPUT is the tick count, "
                         "so inference must multiply it by delta_s (solution_delta / oracle scorer).")
    ap.add_argument("--target-delta-raw-weights", action="store_true",
                    help="B-A: keep the metric weights |clip(y,2)| (raw target) in the MSE term "
                         "instead of the tick weights |clip(k,2)|; corr term is scale-invariant.")
    ap.add_argument("--standardize", action="store_true",
                    help="standardize inputs with feature mean/std from --feat-stats")
    ap.add_argument("--feat-stats", default="/mnt/d/wnn_work/models/feat_stats.npz")
    ap.add_argument("--feat-stats-eng", default="/mnt/d/wnn_work/models/feat_stats_eng.npz",
                    help="normalization stats for --feat-mode eng")
    ap.add_argument("--weighted-loss", action="store_true",
                    help="MSE weighted by |clipped target| (metric-aligned)")
    ap.add_argument("--corr-loss", action="store_true",
                    help="add a metric-aligned block-wise weighted-Pearson term "
                         "(the leaderboard metric itself)")
    ap.add_argument("--corr-unit-mix", type=float, default=0.0,
                    help="ALROWS objective (plan FINAL_7SLOTS A1/A2): mix weight of the "
                         "UNWEIGHTED correlation over ALL rows into the corr term. "
                         "1.0 = pure allrows objective, 0.0 = the old weighted-by-|y| loss.")
    ap.add_argument("--corr-mix", type=float, default=1.0,
                    help="weight of the correlation term when --corr-loss is on "
                         "(1.0 = pure correlation loss, 0.5 = half MSE)")
    ap.add_argument("--level-mix", type=float, default=0.0,
                    help="weight of the per-SEQUENCE LEVEL correlation term in "
                         "the corr loss (0 = off, byte-identical to before)")
    ap.add_argument("--pooled-mix", type=float, default=0.0,
                    help="weight of the GLOBAL (pooled) weighted-Pearson term, "
                         "i.e. the exact platform metric over the batch "
                         "(0 = off, byte-identical to before)")
    ap.add_argument("--atten-mix", type=float, default=0.0,
                    help="weight of the scale-anchored attenuation penalty "
                         "(src/atten_loss.py); 0 disables it")
    ap.add_argument("--scale-mix", type=float, default=0.0,
                    help="weight of the per-sequence output-SHRINKAGE penalty "
                         "(21.10: the hard class halves its output spread, "
                         "hard_r 0.4714; the oracle fix is worth +0.058)")
    ap.add_argument("--corr-horizon", choices=["seg", "seq", "mid", "global"], default="seq",
                    help="window for the correlation term: 'seg' = per seg-len "
                         "segment (noisy, targets are near-constant in short "
                         "windows); 'seq' = accumulated over the whole sequence "
                         "(matches the metric; detached past segments)")
    ap.add_argument("--corr-chunk", type=int, default=0,
                    help="chunk-Pearson: recompute the block correlation every N rows "
                         "and average over chunks (0 = whole sequence, legacy). "
                         "Mirrors the public LOB stack's 1000-row recomputation.")
    ap.add_argument("--corr-pool-chunk", type=int, default=0,
                    help="pooled chunk-Pearson: recompute the POOLED correlation over the "
                         "batch's rows every N steps and average over chunks (0 = off). "
                         "This is the public stack's object; --corr-chunk is per-sequence "
                         "block and is blind to the between-sequence level (drift, F336).")
    ap.add_argument("--dnpl", type=float, default=0.0,
                    help="Dispersion-Normalised PCC mix in [0,1]: weight the "
                         "correlation term by sg(sigma_pred/sigma_y) so that "
                         "shrinking the output stops being a free win. "
                         "0 = off (legacy behaviour, bit-identical).")
    ap.add_argument("--corr-mid-window", type=int, default=0,
                    help="S4: when --corr-horizon mid, correlate over the last N "
                         "steps of the accumulated window instead of all of it "
                         "(0 = unused). 'mid' sits between 'seg' (one segment, "
                         "noisy) and 'seq' (whole sequence) — the untested cell.")
    ap.add_argument("--row-weight-col", default=None,
                    help="per-row weight column to read from the parquet (e.g. "
                         "'is_scored' on valid.parquet): the EXACT form of the "
                         "mask-alignment lever. Read alongside features and used "
                         "as a row multiplier. Default none = unchanged.")
    ap.add_argument("--row-weight-hybrid", type=int, default=None,
                    help="E6 hybrid: leading N row groups carry p_hat weights, "
                         "the rest read the EXACT weight column from "
                         "--row-weight-mask-file (aligned 1:1 with the valid part "
                         "of the combined file). For the standard build pass 10607.")
    ap.add_argument("--exact-oversample", type=int, default=1,
                    help="E6 hybrid step-2b: append each valid-side group K-1 "
                         "extra times to the epoch order (raises the DOSE of the "
                         "exact-mask side; kappa proved to be a no-op). 1 = off.")
    ap.add_argument("--row-weight-train-ones", action="store_true",
                    help="E6 factorial cell: train-side groups get weight 1.0 "
                         "instead of the p_hat proxy (valid side keeps exact).")
    ap.add_argument("--row-weight-exact-kappa", type=float, default=1.0,
                    help="E6 hybrid step-2: multiply the EXACT is_scored side "
                         "weights by this factor (upweight what the metric "
                         "actually scores). 1.0 = original E6 behaviour.")
    ap.add_argument("--row-weight-mask-file", default=None,
                    help="E6 hybrid: parquet with the is_scored column, row groups "
                         "aligned to the valid part of --data (e.g. valid_mask.parquet)")
    ap.add_argument("--row-weight-hard-kappa", type=float, default=1.0,
                    help="A1 (24.10): multiply the EXACT-mask row weights of the "
                         "HARD-carrier groups (delta=0.65687, class 3) by this "
                         "factor. Reads tools/valid_class_seq.json; the hard_idx "
                         "must be 3 (self-check). 1.0 = inert/off. Only with "
                         "--row-weight-hybrid.")
    ap.add_argument("--mask-p", type=float, default=None,
                    help="C2' sampling-robust (24.10): keep each EXACT-side row with "
                         "probability mask_p (FRESH bernoulli draw per row-group per "
                         "epoch): the model no longer sees ONE fixed mask but a "
                         "distribution of sub-masks. N43 'second cell' — never run "
                         "before. None/1.0 = off (byte-identical).")
    ap.add_argument("--drift-mix", type=float, default=0.0,
                    help="penalty weight for prediction movement on rows where the target did "
                         "not move (holding the level). 0 = off.")
    ap.add_argument("--row-weight-abs", action="store_true",
                    help="use |col| for --row-weight-col (required when the column is signed, "
                         "e.g. a target column). Without it negative weights abort.")
    ap.add_argument("--row-weight-floor", type=float, default=0.0,
                    help="lower clamp for --row-weight-col values (default 0 = exact)")
    ap.add_argument("--row-weight-gamma", type=float, default=1.0,
                    help="optional sharpening exponent for --row-weight-col (default 1)")
    ap.add_argument("--change-boost", type=float, default=0.0,
                    help="10.11.2026 CHANGE-BOOST: multiply each row's loss weight by "
                         "(1 + lambda) when t0 changes vs the previous step, 1.0 when it holds. "
                         "Measured on full valid: 92.1%% of scored rows are holds (lag-1 = 0.99977, "
                         "nothing to distinguish), all model differences live on the 7.87%% change "
                         "rows where ours is 0.825 vs 0.850 for lag-1. 0.0 = off (byte-identical).")
    ap.add_argument("--row-weight-mode", choices=["none", "phat"], default="none",
                    help="per-row weight multiplier: 'phat' = predicted P(is_scored) "
                         "from a0..a7 (the platform scores only ~8-12%% of rows and "
                         "they are predictable; see docs/RESEARCH_00_MASK_HEADLINE.md). "
                         "Requires --phat-coeffs. Default none = unchanged behaviour.")
    ap.add_argument("--row-weight-blockeq", action="store_true",
                    help="нормировать веса строк на суммарный вес последовательности: каждый блок "
                         "даёт равный вклад в лосс, как в метрике (метрика усредняет блоки, "
                         "а не строки). Закрывает недопредставленность блоков с редкой маской "
                         "(F285: счёт блока падает с ростом mask_frac, r=-0.362)")
    ap.add_argument("--phat-coeffs", default="/mnt/d/wnn_work/tmp/mask_logit_coeffs.json",
                    help="JSON with {w_raw:[8], b_raw:float} for the p_hat logistic")
    ap.add_argument("--phat-gamma", type=float, default=2.0,
                    help="row_w = clamp(p_hat, floor, 1) ** gamma")
    ap.add_argument("--phat-floor", type=float, default=0.02,
                    help="lower clamp for row_w (keeps gradient on non-scored-like rows)")
    ap.add_argument("--phat-select", choices=["none", "threshold", "topk"], default="none",
                    help="A2 proxy-mask class: HARD selection 1/0 from p_hat. "
                         "'threshold' = 1{p_hat > --phat-threshold}; 'topk' = the "
                         "--phat-topk highest-p_hat rows per sequence. Ignores "
                         "--phat-floor/--phat-gamma (weight is exactly 1 or 0; keeps "
                         "the mask's gradient concentration without the exact-metre "
                         "poison). Only used with --row-weight-mode phat.")
    ap.add_argument("--phat-threshold", type=float, default=None,
                    help="threshold for --phat-select threshold (choose so the "
                         "selected fraction is 8-12%%, like the real mask)")
    ap.add_argument("--phat-topk", type=int, default=None,
                    help="K rows per sequence for --phat-select topk "
                         "(K = 1600..2400 of 20000 covers the 8-12%% mask dose)")
    ap.add_argument("--selftest", action="store_true",
                    help="A2 proxy-mask selftest on 5 synthetic batches (0 GPU, "
                         "0 data): weights in {0,1}, expected fraction, no NaN, "
                         "gradient reaches the loss. Prints numbers and exits.")
    ap.add_argument("--feat-mode", choices=["raw", "eng", "stream336", "delta224", "win"], default="raw",
                    help="raw = 112 given columns; eng = engineered features "
                         "(feat_eng.make_features); stream336 = raw + EMA-28 residual "
                         "+ lag-42 diff (mnemonik champion set, 336 columns). "
                         "Affects model input width.")
    ap.add_argument("--head-raw", action="store_true",
                    help="D1: the linear head sees [hidden, raw tick] instead of "
                         "hidden only — the GRU filters, the head can correct with "
                         "the current row. Width becomes hid + n_features.")
    ap.add_argument("--head-mlp", action="store_true",
                    help="Hybrid head: [hidden, raw tick] -> Linear(hid) -> ReLU -> "
                         "Linear(2), i.e. the 11th-place GRU+MLP branch. Requires "
                         "--head-raw; ONNX-only (the C packer is linear-head only).")
    ap.add_argument("--persist-feat", type=int, default=-1,
                    help="D2: append feature column N (raw, pre-standardization) as an "
                         "extra model input. Rationale: the target autocorr lag1 is "
                         "0.98 and a3 (col 107) lives on the same timescale (its own "
                         "lag1 = 1.00), so the model should be told this persistence "
                         "signal explicitly instead of rediscovering it in h. "
                         "-1 = off (default). Width becomes n_features + 1, and the "
                         "extra column is added INSIDE the graph so the 112-wide "
                         "input contract is unchanged.")
    ap.add_argument("--feat-mix", action="store_true",
                    help="F2 (07.10.2026): zero-init residual feature-mixing bottleneck "
                         "(112->32->112, GELU) AFTER the standardization, BEFORE the GRU. "
                         "At init = exact champion (output layer zero-init).")
    ap.add_argument("--feat-gate", action="store_true",
                    help="N6: feature-axis mixing. The GRU output is multiplied by a "
                         "per-unit gate computed from the current (normalized) feature "
                         "row: g = sigmoid(W_g @ x_t), h' = h * g. Rationale: TLOB "
                         "(dual attention incl. the FEATURE axis) and CVML (explicit "
                         "cross-variate mixing) both report that inter-feature mixing "
                         "is the axis LOB models lack; the plain GRU mixes features "
                         "only once inside weight_ih. Cost measured on 1-thread CPU: "
                         "x1.2-1.6 per row -> prefer a 2-member ensemble. off (default).")
    ap.add_argument("--split-window", type=int, default=0,
                    help="S-b: cut each sequence into consecutive windows of N rows, "
                         "each starting from ZERO hidden. 20000 with --split-window "
                         "10000 yields 2x the independent gradient trajectories on "
                         "the same data (RESEARCH_02 §5). The effective batch becomes "
                         "--batch * (20000/N), so lower --batch by that factor. "
                         "0 = off (default). The tail remainder is dropped.")
    ap.add_argument("--shock-mix", type=float, default=0.0,
                    help="A2 (24.10): dose of the SHOCK (innovation) correlation term: "
                         "loss -= shock_mix * block_wp(P_t - 0.93*P_{t-1}, "
                         "Y_t - 0.93*Y_{t-1}, w=|Y_shock|). The measured reserve: "
                         "shock is in the features at r~0.42 while the model "
                         "tracks it at 0.275; the plain block-WP metric is blind "
                         "to that structure (weights + centring hide the "
                         "innovations). 0 = off (inert, byte-identical).")
    ap.add_argument("--rank-mix", type=float, default=0.0,
                    help="H2: dose of a weighted Spearman (rank) term added alongside "
                         "the Pearson corr term. Rationale: the metric weights rows by "
                         "|clip(y)|, so Pearson is dominated by the |y|~2 tail while the "
                         "metric's mass sits in |y| in [0.2, 1.5]; a rank term asks only "
                         "for the ORDER to be right. 0 = off (default). Framed as rank "
                         "vs value, one variable - the Pearson term is untouched.")
    ap.add_argument("--rank-tau", type=float, default=0.05,
                    help="H2: soft-rank temperature for the Spearman term. Ranks are "
                         "approximated by sum_j sigmoid((x_j - x_i)/tau); smaller tau "
                         "approaches exact integer ranks, larger tau is smoother.")
    ap.add_argument("--target-balance", action="store_true",
                    help="give t0 and t1 equal share of the weighted MSE term. "
                         "Measured: E|clip(t1,2)|=0.688 vs t0 0.843 and t1 has more "
                         "zero-weight rows, so the plain weighted loss under-trains "
                         "t1 while the metric scores both targets equally.")
    ap.add_argument("--aux-mix", type=float, default=0.0,
                    help="P1: weight of the self-supervised NEXT-STEP FEATURE head. The head "
                         "predicts x_{t+aux_lag} from the hidden state at t; it exists only "
                         "during training and is dropped at inference, so the exported graph "
                         "(gru + head) and the C kernel are unchanged. 0 = off (default).")
    ap.add_argument("--aux-lag", type=int, default=1,
                    help="P1: how many steps ahead the aux head predicts (default 1).")
    ap.add_argument("--slow-mix", type=float, default=0.0,
                    help="P2: alpha of the slow hidden state (EMA of h). The head then sees "
                         "[h_fast, h_slow] (2*hid inputs). Causal: the EMA is a left-only "
                         "geometric convolution, no look-ahead. 0 = off (default).")
    ap.add_argument("--slow-eps", type=float, default=1e-4,
                    help="P2: tail cutoff for the EMA kernel (weights below this are dropped).")
    ap.add_argument("--loss-clip", type=float, default=0.0,
                    help="METRIC-ALIGNED: clip targets AND weights to +-C in the MSE term, "
                         "exactly as the metric does (|y| reaches 8.93 in data while the metric "
                         "clips at 2). 0 = off (previous behaviour).")
    ap.add_argument("--class-keep", type=float, default=0.0,
                    help="CLASS-SPECIALIST: train only on sequences whose target tick equals this "
                         "value (min non-zero |y|). Tick cache from mcz217. 0 = off.")
    ap.add_argument("--class-tol", type=float, default=0.02,
                    help="tolerance for --class-keep tick matching.")
    ap.add_argument("--class-ticks", type=str, default="",
                    help="npz with per-group ticks (mcz217 output).")
    ap.add_argument("--feat-drop", type=str, default="",
                    help="D-05: comma list of feature groups to DROP, from i0p,i0v,i1p,i1v "
                         "(28 columns each). Makes structurally different members for a blend; "
                         "packaging needs no kernel change (zeroed W_ih columns are equivalent).")
    ap.add_argument("--lag-attn", type=float, default=0.0,
                    help="P4: dose of the LAG-AXIS attention gate. Per group of 28 lags the "
                         "model scores each lag, softmaxes over the 28 and rescales the row "
                         "(mean gate = 1 at uniform attention). Width unchanged, so the kernel "
                         "needs only a block-softmax. 0 = off (default).")
    ap.add_argument("--roll-norm", type=int, default=0, metavar="W",
                    help="R2 line B: replace the global feat_stats standardization with a "
                         "CAUSAL rolling z-score over the last W rows (inclusive). 0 = off. "
                         "Pair with --standardize OFF; the exported graph then expects an "
                         "already-normalized row, which solution.py provides.")
    ap.add_argument("--u-basis", action="store_true",
                    help="TZ-6: train the heads in the u-basis (u+ = (t0+t1)/sqrt2, "
                         "u- = (t0-t1)/sqrt2). The corr term then runs per-column, i.e. "
                         "each mode is normalised by its OWN weighted variance, so the "
                         "u- head stops eclipsing u+ (c3 su_eff=0.0399). The MSE anchor "
                         "stays in (t0,t1); the export rotates back inside the graph.")
    ap.add_argument("--k-head", action="store_true",
                    help="26.10: distributional head over the INTEGER tick count k "
                         "(y = k*tick_c, F085/F089). Output is blended "
                         "pred + lambda*(E[k]*tick_hat - pred), where tick_hat comes from a "
                         "4-way class head. Rationale: the regression shrinks -- on the record "
                         "dump |pred|>=1.6 in 0.91%% of values vs 17.26%% of the target, and HALF "
                         "of the c3 metric weight is on k=3 rows the model never reaches. "
                         "Post-hoc snapping of an existing regression FAILS (F087, -0.0230, "
                         "t -6.28): the level must be learned. Default OFF = bit-identical.")
    ap.add_argument("--k-head-lambda", type=float, default=0.5,
                    help="blend weight of the E[k]*tick decode (default 0.5).")
    ap.add_argument("--k-head-ce", type=float, default=0.3,
                    help="weight of the auxiliary cross-entropy over k (default 0.3).")
    ap.add_argument("--gamma-gate", action="store_true",
                    help="TZ-6: extra scalar gate gamma = sigmoid(w.h_t + b) in (0,1) "
                         "applied to u+ only; p0=(g*u+ + u-)/sqrt2. Zero-initialised "
                         "=> gamma == 0.5 exactly at step 0. Requires --u-basis.")
    args = ap.parse_args()

    if args.gamma_gate and not args.u_basis:
        raise SystemExit("--gamma-gate требует --u-basis (гейт умножает u+, а не t0/t1)")

    if args.selftest:
        ok = _proxymask_selftest("cpu")
        sys.exit(0 if ok else 1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        # Замер 28.09 на этой машине (tools/bench_io_gpu.py, тот же код, что тут):
        # benchmark=False -> 63 мс/посл (16.0 посл/с), benchmark=True -> 9 мс/посл
        # (116.7 посл/с) = 7.3x. Математика одинаковая, меняется только выбор
        # ядра cuDNN; 16 посл/с ровно совпадали с реальными логами обучения, то
        # есть эпохи тормозил не диск, а отключённый автоподбор.
        # Детерминизм нужен только для побитового воспроизведения ранее
        # сохранённых прогонов/чисел: WNN_DETERMINISM=1 возвращает старое
        # поведение (deterministic=True, benchmark=False).
        _det = os.environ.get("WNN_DETERMINISM", "0") == "1"
        torch.backends.cudnn.deterministic = _det
        torch.backends.cudnn.benchmark = not _det
        print(f"[train] cudnn.deterministic={_det} benchmark={not _det} "
              + ("(WNN_DETERMINISM=1: bit-exact, slow)" if _det
                 else "(fast; set WNN_DETERMINISM=1 for bit-exact)"), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(args.data)
    n_groups = pf.num_row_groups
    # A1 hard-carrier dose: build the per-row-group kappa map before any batch.
    _install_hard_kappa(args, pf, n_groups)
    schema_names = pf.schema_arrow.names
    need_cols = list(FEATURE_COLUMNS) + list(TARGET_COLUMNS)
    missing = [c for c in need_cols if c not in schema_names]
    if missing:
        raise SystemExit(f"missing columns in {args.data}: {missing[:5]}...")
    print(f"[train] data={args.data} groups={n_groups} seq_len={SEQUENCE_LENGTH} "
          f"device={device} hid={args.hid} layers={args.layers} batch={args.batch}")

    # input width: with --feat-mode eng the transform lives INSIDE the model
    # graph (FeatureTransform), so the reader always yields the raw 112 columns
    # and the exported ONNX keeps the baseline's 112-wide input contract.
    n_in = N_FEATURES
    use_eng = args.feat_mode == "eng"
    if args.feat_mode == "stream336":
        from stream336 import N_STREAM
        n_in = N_STREAM
    if args.feat_mode == "delta224":
        from delta224 import N_DELTA
        n_in = N_DELTA
    if args.feat_mode == "win":
        from winfeat import N_WIN
        n_in = N_WIN
    if args.roll_norm:
        print(f"[train] ROLL-NORM ON: W={args.roll_norm} (causal window, global stats IGNORED)",
              flush=True)
    print(f"[train] feat_mode={args.feat_mode} in-graph-transform={use_eng} "
          f"gru_input={N_DERIVED_ENG if use_eng else n_in}", flush=True)

    model = FullSeqModel(hid=args.hid, layers=args.layers, dropout=args.dropout,
                         n_features=n_in, use_eng=use_eng,
                         head_raw=args.head_raw, head_mlp=args.head_mlp,
                         persist_col=args.persist_feat,
                         feat_gate=args.feat_gate,
                         feat_mix=args.feat_mix,
                         u_basis=args.u_basis,
                         k_head=args.k_head, k_lambda=args.k_head_lambda,
                         gamma_gate=args.gamma_gate,
                         aux_mix=getattr(args, "aux_mix", 0.0),
                         aux_lag=getattr(args, "aux_lag", 1),
                         slow_mix=getattr(args, "slow_mix", 0.0),
                         slow_eps=getattr(args, "slow_eps", 1e-4),
                         lag_attn=getattr(args, "lag_attn", 0.0),
                         feat_drop=getattr(args, "feat_drop", ""),
                         front_dim=getattr(args, "front_proj", 0) or 0,
                         cell=args.cell_type,
                         hf_features=args.hf_features,
                         hf_causal=args.hf_causal_mod).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] params={n_params} head_raw={args.head_raw} "
          f"persist_feat={args.persist_feat} feat_gate={args.feat_gate} "
          f"u_basis={args.u_basis} gamma_gate={args.gamma_gate} "
          f"gru_input={model.n_features}")
    if args.init:
        init_ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(init_ckpt["state_dict"])
        print(f"[train] warm-start from {args.init} (epoch {init_ckpt.get('epoch')})", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    mean_t = std_t = None
    if args.standardize:
        stats = np.load(args.feat_stats if args.feat_mode == "raw" else args.feat_stats_eng)
        mean_t = torch.from_numpy(stats["mean"].astype(np.float32)).to(device)
        std_t = torch.from_numpy(stats["std"].astype(np.float32)).to(device)
        print(f"[train] standardize ON (stats from "
              f"{args.feat_stats if args.feat_mode == 'raw' else args.feat_stats_eng}, "
              f"{stats['std'].min():.3f}..{stats['std'].max():.3f})", flush=True)
    if args.target_standardize:
        print("[train] TARGET-STANDARDIZE ON (y'=(y-m_s)/sigma_s per sequence, w=|y|)", flush=True)
    if getattr(args, 'target_delta', False):
        print("[train] TARGET-DELTA ON (k=y/delta_s per sequence, w=|clip(k,2)|)", flush=True)
    if args.weighted_loss:
        print("[train] weighted loss ON (w=|clip(y,2)|)", flush=True)
    if args.row_weight_col:
        print(f"[train] row-weight column ON: {args.row_weight_col} "
              f"floor={args.row_weight_floor} gamma={args.row_weight_gamma} "
              f"hybrid_split={args.row_weight_hybrid}", flush=True)
    global _PHAT_W_NP, _PHAT_B_NP
    # A1 hard-carrier dose: build the per-row-group kappa map BEFORE any batch.
    def _len_valid_groups(path):
        return pq.ParquetFile(path).num_row_groups
    if getattr(args, "row_weight_hard_kappa", 1.0) != 1.0:
        if args.row_weight_hybrid is None:
            raise SystemExit("--row-weight-hard-kappa requires --row-weight-hybrid")
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                          "..", "tools_class"))
        import classmap as _cm
        _cm_obj = _cm.load()
        assert _cm_obj.hard_idx == 3, f"hard_idx {_cm_obj.hard_idx} != 3 — labels moved?"
        # The combined file's valid side starts at group row_weight_hybrid; the
        # classmap indexes the FULL valid (1873 seqs) which IS the valid side 1:1.
        n_valid_groups = _len_valid_groups(args.data) - args.row_weight_hybrid
        assert n_valid_groups == len(_cm_obj.class_seq), \
            f"valid groups {n_valid_groups} != classmap {len(_cm_obj.class_seq)}"
        globals()["_HARD_KAPPA"] = {i: float(args.row_weight_hard_kappa)
                                    for i, c in enumerate(_cm_obj.class_seq) if c == 3}
        print(f"[train] A1 hard-kappa ON: {len(globals()['_HARD_KAPPA'])} hard groups "
              f"×{args.row_weight_hard_kappa}", flush=True)
    if args.row_weight_col and args.row_weight_hybrid is not None:
        if not args.row_weight_mask_file:
            raise SystemExit("--row-weight-hybrid requires --row-weight-mask-file")
        import json as _json
        with open(args.phat_coeffs) as _fh:
            _c = _json.load(_fh)
        _PHAT_W_NP = np.asarray(_c["w_raw"], dtype=np.float32)
        _PHAT_B_NP = float(_c["b_raw"])
        _mpf = pq.ParquetFile(args.row_weight_mask_file)
        print(f"[train] hybrid E6 ON: exact-kappa={args.row_weight_exact_kappa} "
              f"groups < {args.row_weight_hybrid} get p_hat "
              f"({len(_PHAT_W_NP)} feats), rest get '{args.row_weight_col}' from "
              f"{args.row_weight_mask_file} ({_mpf.num_row_groups} groups)", flush=True)
    global _PHAT_W, _PHAT_B, _PHAT_NF
    if args.row_weight_mode == "phat":
        args.phat_coeffs = _phat_coeffs_path(args.phat_coeffs)
        _PHAT_W, _PHAT_B, _PHAT_NF = _load_phat_coeffs(args.phat_coeffs)
        _PHAT_W = _PHAT_W.to(device)
        _PHAT_B = torch.tensor(_PHAT_B, dtype=torch.float32, device=device)
        print(f"[train] row-weight mode phat ON: coeffs={args.phat_coeffs} "
              f"n_features={_PHAT_NF} gamma={args.phat_gamma} "
              f"floor={args.phat_floor}", flush=True)

    teachers = None
    if args.kd_ckpts and args.kd_alpha > 0:
        teachers = []
        for tp in args.kd_ckpts:
            tck = torch.load(tp, map_location="cpu", weights_only=False)
            tm = FullSeqModel(hid=tck["hid"], layers=tck["layers"])
            tm.load_state_dict(tck["state_dict"])
            tm.eval().to(device)
            for p in tm.parameters():
                p.requires_grad_(False)
            teachers.append(tm)
            print(f"[train] KD teacher: {tp} (hid={tck['hid']})", flush=True)
        print(f"[train] KD ON: alpha={args.kd_alpha}, teachers={len(teachers)}", flush=True)

    log = {
        "config": vars(args), "n_groups": n_groups, "params": n_params,
        "device": device, "torch": torch.__version__, "steps": [],
        "epoch_loss": [], "epoch_time_s": [],
    }
    log_path = out_dir / f"{args.tag}_train_log.json"
    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    for epoch in range(args.epochs):
        # C2' sampling-robust: a FRESH RNG seed per epoch for the mask thinning
        globals()["_MASK_RNG"] = (args.seed * 100003 + epoch) & 0xFFFFFFFF
        model.train()
        if device == "cuda":
            torch.cuda.empty_cache()
        if args.lr_decay == "cosine":
            prog = epoch / max(1, args.epochs - 1)
            lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * prog))
            for g in opt.param_groups:
                g["lr"] = lr
            print(f"[train] epoch {epoch} lr={lr:.2e}", flush=True)
        elif args.lr_decay == "linear":
            # Non-cosine ANNIHILATION (21.10, программа S клетка S3).
            # Why: the cosine class buys locally (+0.005..+0.006 per member) and
            # PAYS a transfer tax on the platform (-0.007..-0.018, see
            # docs/N32_ENGINE_TAX_VERDICT.md 15.10). The non-cosine schedules
            # were never tried at all, and they reach the same end-of-training
            # lr as cosine WITHOUT the cosine shape - so if the local gain
            # comes from the small final steps and not from the cosine curve,
            # linear keeps the gain and drops the tax.
            prog = epoch / max(1, args.epochs - 1)
            lr = args.min_lr + (args.lr - args.min_lr) * (1.0 - prog)
            for g in opt.param_groups:
                g["lr"] = lr
            print(f"[train] epoch {epoch} lr={lr:.2e} (linear decay)", flush=True)
        ep_loss, ep_steps, ep_rows = 0.0, 0, 0
        ep_atten = {}
        t_ep = time.time()
        order = epoch_group_order(n_groups, args.max_seq, rng,
                                  exact_split=args.row_weight_hybrid,
                                  oversample=getattr(args, "exact_oversample", 1),
                                  start=getattr(args, "seq_start", 0) or 0,
                                  group_stride=getattr(args, "group_stride", 0) or 0,
                                  group_offset=getattr(args, "group_offset", 0) or 0,
                                  group_keep=(load_group_list(args.group_list_file)
                                              if getattr(args, "group_list_file", None) else None))
        # CLASS-SPECIALIST (05.11): отбор последовательностей по тику цели.
        # Тик = min ненулевой |y| (F089), кэш строится mcz217. Метрика взвешивает строки
        # по |y|, и сложный класс (0.657) даёт 42% веса при корреляции 0.50 против 0.78
        # у лёгких — значит запас лежит именно там, и специалист его проверит.
        _ck = float(getattr(args, "class_keep", 0.0) or 0.0)
        if _ck > 0:
            _tick_file = getattr(args, "class_ticks", "")
            _ct = np.load(_tick_file)["ticks"] if _tick_file else None
            if _ct is None or len(_ct) <= max(order):
                raise SystemExit(f"--class-keep: нет кэша тиков для {max(order)+1} групп "
                                 f"(получено {None if _ct is None else len(_ct)})")
            _tol = float(getattr(args, "class_tol", 0.02) or 0.02)
            order = [g for g in order if abs(float(_ct[g]) - _ck) <= _tol]
            print(f"[train] class-keep {_ck} +-{_tol}: {len(order)} последовательностей", flush=True)
            if not order:
                raise SystemExit("--class-keep: не осталось ни одной последовательности")
        batch_iter = batches_in_order(order, args.batch, rng)
        loader = prefetch(batch_iter,
                          lambda ids: read_batch(args.data, ids, need_cols,
                                                 use_threads=args.reader_use_threads,
                                                 feat_mode=args.feat_mode,
                                                 weight_col=args.row_weight_col,
                                                 hybrid_split=args.row_weight_hybrid,
                                                 mask_path=args.row_weight_mask_file,
                                                 exact_kappa=args.row_weight_exact_kappa,
                                                 train_ones=getattr(args, "row_weight_train_ones", False),
                                                 mask_p=getattr(args, "mask_p", None)),
                          workers=args.prefetch_workers)
        for step, batch in enumerate(loader):
            if len(batch) == 3:
                feats, targs, wrow = batch
            else:
                feats, targs = batch
                wrow = None
            try:
                if getattr(args, "split_window", 0):
                    feats, targs, wrow = split_windows(feats, targs, wrow,
                                                       args.split_window)
                loss_val, n_rows, info = train_step(model, opt, feats, targs, args, device, mean_t, std_t,
                                              teachers=teachers, kd_alpha=args.kd_alpha,
                                              row_mask=wrow)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                half = max(1, feats.shape[0] // 2)
                print(f"[train] CUDA OOM at batch={feats.shape[0]}; retrying in halves of {half}", flush=True)
                loss_val, n_rows, info = 0.0, 0, {}
                for s in range(0, feats.shape[0], half):
                    lv, nr, ifo = train_step(model, opt, feats[s:s + half], targs[s:s + half], args, device,
                                        mean_t, std_t, teachers=teachers, kd_alpha=args.kd_alpha,
                                        row_mask=None if wrow is None else wrow[s:s + half])
                    loss_val += lv
                    n_rows += nr
                    info = ifo

            ep_loss += loss_val
            ep_steps += 1
            ep_rows += n_rows
            if "atten" in info:
                for _k, _v in info["atten"].items():
                    ep_atten.setdefault(_k, []).append(_v)
            if step % args.log_every == 0:
                speed = ep_rows / max(time.time() - t_ep, 1e-6) / SEQUENCE_LENGTH
                vram = torch.cuda.memory_allocated() / 2**20 if device == "cuda" else 0
                mem = mem_snapshot()
                rho_txt = f" rho={info['rho']:+.4f}" if "rho" in info else ""
                rk_txt = f" rank={info['rank']:+.4f}" if "rank" in info else ""
                mse_txt = f" mse={info['mse']:.4f}" if "mse" in info else ""
                print(f"[train] e{epoch} step {step} loss={loss_val:.6f}{rho_txt}{rk_txt}{mse_txt} "
                      f"seq/s={speed:.1f} vram={vram:.0f}MB "
                      f"cached={mem.get('Cached', -1)}MB avail={mem.get('MemAvailable', -1)}MB", flush=True)
                _step_rec = {"epoch": epoch, "step": step, "loss": loss_val,
                             "rho": info.get("rho"), "mse": info.get("mse"),
                             "seq_s": speed, "vram_mb": vram, "mem": mem}
                if "atten" in info:
                    _step_rec["atten"] = info["atten"]
                log["steps"].append(_step_rec)
                with open(log_path, "w") as fh:
                    json.dump(log, fh, indent=1)

        ep_time = time.time() - t_ep
        mean_loss = ep_loss / max(ep_steps, 1)
        log["epoch_loss"].append(mean_loss)
        log["epoch_time_s"].append(ep_time)
        if ep_atten:
            log.setdefault("atten_epoch", []).append(
                {k: sum(v) / max(len(v), 1) for k, v in ep_atten.items()})
        _atxt = ""
        if ep_atten:
            _a = {k: sum(v) / max(len(v), 1) for k, v in ep_atten.items()}
            _atxt = ("  | atten alpha=%.4f |1-a|=%.4f spread=%.4f pen=%.4f noise=%.4f"
                     % (_a.get("mean_alpha", 0), _a.get("mean_abs_1_minus_alpha", 0),
                        _a.get("mean_spread_ratio", 0), _a.get("mean_atten_penalty", 0),
                        _a.get("mean_noise_excess", 0)))
        print(f"[train] EPOCH {epoch} done: mean_loss={mean_loss:.6f} time={ep_time:.0f}s "
              f"({ep_rows/SEQUENCE_LENGTH/max(ep_time,1e-6):.1f} seq/s){_atxt}", flush=True)

        ckpt = {"state_dict": model.state_dict(), "hid": args.hid, "layers": args.layers,
                "epoch": epoch, "config": vars(args),
                "norm_mean": None if mean_t is None else mean_t.cpu().numpy(),
                "norm_std": None if std_t is None else std_t.cpu().numpy()}
        torch.save(ckpt, out_dir / f"{args.tag}_e{epoch}.pt")
        with open(log_path, "w") as fh:
            json.dump(log, fh, indent=1)

    torch.save({"state_dict": model.state_dict(), "hid": args.hid, "layers": args.layers,
                "config": vars(args),
                "norm_mean": None if mean_t is None else mean_t.cpu().numpy(),
                "norm_std": None if std_t is None else std_t.cpu().numpy()}, out_dir / f"{args.tag}.pt")
    model.eval().cpu()
    if mean_t is not None:
        export_onnx_norm(model, out_dir / f"{args.tag}.onnx", args.hid, args.layers,
                         mean_t.cpu().numpy(), std_t.cpu().numpy())
    else:
        export_onnx(model, out_dir / f"{args.tag}.onnx", args.hid, args.layers)
    log["total_time_s"] = time.time() - t_start
    with open(log_path, "w") as fh:
        json.dump(log, fh, indent=1)
    print(f"[train] DONE total={log['total_time_s']:.0f}s onnx={out_dir / (args.tag + '.onnx')}")


if __name__ == "__main__":
    main()
