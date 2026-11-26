"""Knowledge-distillation primitives for the WNN33 regression cells.

The competitor's cascade (research/ensdistill.py) uses plain MSE imitation, not
KL: for every row the soft target is the weighted mean of several teacher
members, and the student minimises

    loss = (1 - alpha) * label + alpha * mimic

where label and mimic are |y|-weighted MSE against the labels and the soft
target respectively.  The mask is the true ``is_scored`` column on validation
rows (unscored rows get a small floor, 0.06 in their recipe) and a mask-net
expectation on train rows.  This module keeps the same semantics without
depending on their code.

Public API:
    kd_targets_from_ensemble(preds_list, weights=None) -> np.ndarray
    kd_loss(student_pred, teacher_pred, alpha, mask=None) -> torch.Tensor
    build_soft_targets_from_checkpoints(checkpoints, raw_features, ...) -> np.ndarray
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def kd_targets_from_ensemble(preds_list: Sequence, weights: Sequence[float] | None = None) -> np.ndarray:
    """Return the weighted mean of several member predictions.

    ``preds_list`` is a sequence of arrays/tensors with identical shapes
    ``(..., 2)``.  ``weights`` defaults to equal weights and is normalised to
    sum to one.  Two identical members therefore reproduce that member exactly.
    """
    if preds_list is None or len(preds_list) == 0:
        raise ValueError("preds_list must contain at least one member prediction")
    preds = [_to_numpy(p).astype(np.float32, copy=False) for p in preds_list]
    shape = preds[0].shape
    for i, p in enumerate(preds):
        if p.shape != shape:
            raise ValueError(f"member {i} has shape {p.shape}, expected {shape}")
    if weights is None:
        w = np.full(len(preds), 1.0 / len(preds), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or len(w) != len(preds):
            raise ValueError(f"weights must have length {len(preds)}, got {w.shape}")
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite and non-negative")
        total = w.sum()
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        w = w / total
    out = np.zeros(shape, dtype=np.float64)
    for wi, p in zip(w, preds):
        out += wi * p
    return out.astype(np.float32)


def _broadcast_mask(mask, like: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(mask):
        m = mask.to(device=like.device, dtype=like.dtype)
    else:
        m = torch.as_tensor(mask, device=like.device, dtype=like.dtype)
    if m.shape != like.shape:
        if m.ndim == like.ndim - 1:
            m = m.unsqueeze(-1)
        try:
            m = torch.broadcast_to(m, like.shape)
        except RuntimeError as exc:
            raise ValueError(f"mask shape {tuple(m.shape)} is not broadcastable to {tuple(like.shape)}") from exc
    return m


def weighted_mse(pred, target, weight=None) -> torch.Tensor:
    """Mask/weight-aware MSE with a normalised denominator (0 when weight=0)."""
    pred = torch.as_tensor(pred, dtype=torch.float32) if not torch.is_tensor(pred) else pred
    target = torch.as_tensor(target, dtype=pred.dtype, device=pred.device) if not torch.is_tensor(target) else target.to(pred)
    sq = (pred - target).square()
    if weight is None:
        return sq.mean()
    w = _broadcast_mask(weight, sq)
    denom = w.sum()
    if float(denom.detach()) == 0.0:
        return pred.sum() * 0.0
    return (sq * w).sum() / denom


def kd_loss(student_pred, teacher_pred, alpha: float, mask=None) -> torch.Tensor:
    """Alpha-weighted masked MSE between the student and the teacher.

    ``kd_loss(..., alpha=1)`` is exactly the (masked) MSE to the teacher;
    ``alpha=0`` removes the KD contribution.  With ``teacher_pred ==
    student_pred`` the result is zero.  ``mask`` may be a boolean/float tensor
    shaped ``(B, T)`` or ``(B, T, 1)``; rows with zero weight are ignored.
    """
    student = student_pred if torch.is_tensor(student_pred) else torch.as_tensor(student_pred, dtype=torch.float32)
    if not torch.is_tensor(teacher_pred):
        teacher = torch.as_tensor(teacher_pred, dtype=student.dtype, device=student.device)
    else:
        teacher = teacher_pred.to(device=student.device, dtype=student.dtype)
    if student.shape != teacher.shape:
        raise ValueError(f"student shape {tuple(student.shape)} != teacher shape {tuple(teacher.shape)}")
    return float(alpha) * weighted_mse(student, teacher, mask)


class _GRUModel(torch.nn.Module):
    def __init__(self, n_features: int, hid: int, layers: int):
        super().__init__()
        self.gru = torch.nn.GRU(n_features, hid, num_layers=layers, batch_first=True)
        self.head = torch.nn.Linear(hid, 2)

    def forward(self, x, hidden=None):
        out, hidden = self.gru(x, hidden)
        return self.head(out), hidden


class _LSTMModel(torch.nn.Module):
    def __init__(self, n_features: int, hid: int, layers: int):
        super().__init__()
        self.rnn = torch.nn.LSTM(n_features, hid, num_layers=layers, batch_first=True)
        self.head = torch.nn.Linear(hid, 2)

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        return self.head(out), hidden


class _TwinModel(torch.nn.Module):
    def __init__(self, width: int, mode: str = "twinenc"):
        super().__init__()
        from twin_arch import TwinEncGRU

        self.twin = TwinEncGRU(width, mode=mode)

    def forward(self, x, hidden=None):
        return self.twin(x, hidden)


def load_checkpoint_model(path: str | Path, device: str = "cpu"):
    """Build the model matching a training checkpoint and load its weights.

    Supports the plain GRU / LSTM / twin-encoder state dicts written by
    ``src/train.py``.  Returns ``(model, checkpoint_dict)``.
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(sd, dict):
        raise ValueError(f"{path}: no state_dict found")
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(cfg, dict):
        cfg = {}

    if any(k.startswith("twin.") for k in sd):
        width = int(sd["twin.head.weight"].shape[1])
        mode = cfg.get("cell_type", "twinenc")
        if mode not in ("twin", "twinenc"):
            mode = "twinenc" if any(k.startswith("twin.encoders.") for k in sd) else "twin"
        # Внешний head FullSeqModel в twin-ветке НЕ используется: forward
        # возвращает out = self.twin(x), где своя голова уже внутри TwinEncGRU
        # (ключи twin.head.*). Поэтому внешние head.* — лишние, и их надо
        # отбросить, иначе load_state_dict ругается "Unexpected key(s)".
        sd = {k: v for k, v in sd.items()
              if k.startswith("twin.") or k not in ("head.weight", "head.bias")}
        model = _TwinModel(width, mode=mode)
    elif "gru.weight_ih_l0" in sd:
        hid = int(sd["gru.weight_hh_l0"].shape[1])
        layers = sum(1 for k in sd if k.startswith("gru.weight_hh_l"))
        n_features = int(sd["gru.weight_ih_l0"].shape[1])
        model = _GRUModel(n_features, hid, layers)
    elif "rnn.weight_ih_l0" in sd:
        hid = int(sd["rnn.weight_hh_l0"].shape[1])
        layers = sum(1 for k in sd if k.startswith("rnn.weight_hh_l"))
        n_features = int(sd["rnn.weight_ih_l0"].shape[1])
        model = _LSTMModel(n_features, hid, layers)
    else:
        raise ValueError(f"{path}: unsupported state dict keys {list(sd)[:8]}")

    model.load_state_dict(sd)
    model.to(device).eval()
    return model, ckpt


def _normalizer(ckpt, n_features: int):
    if not isinstance(ckpt, dict):
        return None, None
    mean, std = ckpt.get("norm_mean"), ckpt.get("norm_std")
    if mean is None or std is None:
        return None, None
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    if mean.shape[0] != n_features or std.shape[0] != n_features:
        raise ValueError(f"normalizer width {mean.shape[0]}/{std.shape[0]} != {n_features}")
    return mean, std


@torch.no_grad()
def predict_checkpoint(model, raw_features, mean=None, std=None,
                       batch_size: int = 16, device: str = "cpu", dtype=np.float32) -> np.ndarray:
    """Run one checkpoint over ``raw_features`` shaped ``(N, T, F)`` (or ``(B, N, T, F)``)."""
    x = np.asarray(raw_features, dtype=np.float32)
    if mean is not None:
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        x = (x - mean) / std
    model.eval().to(device)
    out = []
    for start in range(0, x.shape[0], batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(x[start:start + batch_size])).to(device)
        pred, _ = model(xb)
        out.append(pred.detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(dtype, copy=False)


def build_soft_targets_from_checkpoints(checkpoint_paths: Iterable[str | Path],
                                        raw_features,
                                        weights: Sequence[float] | None = None,
                                        out_path: str | Path | None = None,
                                        group_ids: Sequence[int] | None = None,
                                        batch_size: int = 16,
                                        device: str = "cpu",
                                        member_dtype=np.float16) -> np.ndarray:
    """Build teacher soft targets from N checkpoints and optionally save an .npz.

    ``raw_features`` is the raw (unnormalised) input for every sequence; each
    checkpoint is normalised with the mean/std stored in that checkpoint, so
    members trained with different statistics can be mixed.  The .npz stores
    ``targets`` (float32, N x T x 2), ``member_preds`` (float16), ``weights``,
    ``member_paths`` and ``group_ids`` when supplied.
    """
    paths = [str(p) for p in checkpoint_paths]
    if not paths:
        raise ValueError("checkpoint_paths must contain at least one checkpoint")
    member_preds = []
    for path in paths:
        model, ckpt = load_checkpoint_model(path, device=device)
        n_features = int(np.asarray(raw_features).shape[-1])
        mean, std = _normalizer(ckpt, n_features)
        member_preds.append(predict_checkpoint(model, raw_features, mean, std,
                                               batch_size=batch_size, device=device))
        del model
    targets = kd_targets_from_ensemble(member_preds, weights)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        w = (np.full(len(paths), 1.0 / len(paths), dtype=np.float64) if weights is None
             else np.asarray(weights, dtype=np.float64))
        w = w / w.sum()
        payload = {
            "targets": targets,
            "member_preds": np.stack(member_preds).astype(member_dtype),
            "weights": w,
            "member_paths": np.asarray(paths),
        }
        if group_ids is not None:
            payload["group_ids"] = np.asarray(group_ids, dtype=np.int64)
        np.savez_compressed(out_path, **payload)
    return targets
