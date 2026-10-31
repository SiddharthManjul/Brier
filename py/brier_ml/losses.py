from __future__ import annotations

import math
from numbers import Real

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_auc_score


def _log_probs(opt_logits: torch.Tensor, valid: torch.Tensor, temperature: float) -> torch.Tensor:
    if (not isinstance(opt_logits, torch.Tensor) or not isinstance(valid, torch.Tensor)
            or opt_logits.ndim != 2 or opt_logits.shape[1] != 6 or not opt_logits.shape[0]
            or not opt_logits.is_floating_point() or not torch.isfinite(opt_logits).all()
            or valid.shape != opt_logits.shape or valid.dtype != torch.bool
            or valid.device != opt_logits.device
            or not torch.isin(valid.sum(-1), torch.tensor([2, 5, 6], device=valid.device)).all()):
        raise ValueError("logits must be finite [B, 6] floats with boolean masks containing K=2, 5 or 6")
    if isinstance(temperature, bool) or not isinstance(temperature, Real) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    logits = opt_logits.float() if opt_logits.dtype in {torch.float16, torch.bfloat16} else opt_logits
    scaled = logits / temperature
    if not torch.isfinite(scaled[valid]).all():
        raise ValueError("temperature produces nonfinite valid logits")
    result = torch.log_softmax(scaled.masked_fill(~valid, -torch.inf), dim=-1)
    if not torch.isfinite(result[valid]).all():
        raise ValueError("valid log probabilities exceed the floating-point range")
    return result


def option_probs(opt_logits: torch.Tensor, valid: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Normalize valid answer logits; invalid positions have exactly zero probability."""
    return _log_probs(opt_logits, valid, temperature).exp()


def proper_loss(opt_logits: torch.Tensor, valid: torch.Tensor, label_idx: torch.Tensor,
                question_is_score: torch.Tensor, ordinal_map: torch.Tensor,
                lam: float = 1.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Average CE plus Brier, or normalized RPS using a position-to-ordinal map."""
    log_p = _log_probs(opt_logits, valid, 1.0)
    b, device = opt_logits.shape[0], opt_logits.device
    for value, shape, dtype in ((label_idx, (b,), torch.long), (question_is_score, (b,), torch.bool),
                                (ordinal_map, (b, 6), torch.long)):
        if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype or value.device != device:
            raise ValueError("labels, score flags and ordinal maps must have matching shapes, dtypes and device")
    if isinstance(lam, bool) or not isinstance(lam, Real) or not math.isfinite(lam) or lam < 0:
        raise ValueError("lam must be finite and nonnegative")
    if ((label_idx < 0) | (label_idx >= 6)).any() or not valid.gather(1, label_idx[:, None]).all():
        raise ValueError("each label must identify a valid option")
    count, positions = valid.sum(-1), torch.arange(6, device=device)[None, :]
    mapped = ordinal_map.masked_fill(~valid, 6)
    expected = torch.where(positions < count[:, None], positions, 6)
    if ((question_is_score & (count != 5)).any() or (ordinal_map[~valid] != -1).any()
            or (ordinal_map[~question_is_score] != -1).any()
            or not torch.equal(mapped.sort(-1).values[question_is_score], expected[question_is_score])):
        raise ValueError("score rows require K=5 and an ordinal permutation; unused map positions must be -1")
    p, onehot = log_p.exp(), F.one_hot(label_idx, 6).to(opt_logits.dtype)
    ce = -log_p.gather(1, label_idx[:, None]).squeeze(1)
    squared = ((p - onehot) ** 2 * valid).sum(-1)
    order = mapped.argsort(-1)
    cumulative = (p.gather(1, order) - onehot.gather(1, order)).cumsum(-1)
    ranked = (cumulative.square() * (positions < count[:, None] - 1)).sum(-1) / (count - 1)
    extra = torch.where(question_is_score, ranked, squared)
    return (ce + lam * extra).mean(), {"ce": ce.mean(), "extra": extra.mean()}


def _multiclass(p: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    probabilities, labels = np.asarray(p, dtype=np.float64), np.asarray(y)
    if (probabilities.ndim != 2 or not len(probabilities) or probabilities.shape[1] not in {2, 5, 6}
            or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1))
            or not np.allclose(probabilities.sum(-1), 1, rtol=0, atol=1e-6)
            or labels.shape != (len(probabilities),) or not np.issubdtype(labels.dtype, np.integer)
            or np.any((labels < 0) | (labels >= probabilities.shape[1]))):
        raise ValueError("expected normalized finite [N, K] probabilities and integer labels in [0, K)")
    return probabilities, labels.astype(np.int64)


def nll(p: ArrayLike, y: ArrayLike) -> float:
    """Mean negative log likelihood, clipping zero probabilities at 1e-12."""
    probabilities, labels = _multiclass(p, y)
    return float(-np.log(probabilities[np.arange(len(labels)), labels].clip(1e-12)).mean())


def brier(p: ArrayLike, y: ArrayLike) -> float:
    """Mean multiclass sum of squared probability errors."""
    probabilities, labels = _multiclass(p, y)
    return float(np.square(probabilities - np.eye(probabilities.shape[1])[labels]).sum(-1).mean())


def rps(p: ArrayLike, y: ArrayLike) -> float:
    """Normalized ranked probability score; columns and labels must be in ordinal order."""
    probabilities, labels = _multiclass(p, y)
    cumulative = (probabilities - np.eye(probabilities.shape[1])[labels]).cumsum(-1)[:, :-1]
    return float(np.square(cumulative).mean())


def accuracy(p: ArrayLike, y: ArrayLike) -> float:
    """Fraction of labels equal to the most probable option, breaking ties by first position."""
    probabilities, labels = _multiclass(p, y)
    return float((probabilities.argmax(-1) == labels).mean())


def _binary(p: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    probabilities, labels = np.asarray(p, dtype=np.float64), np.asarray(y)
    if (probabilities.ndim != 1 or not len(probabilities) or labels.shape != probabilities.shape
            or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1))
            or not np.isin(labels, [0, 1]).all()):
        raise ValueError("expected matching nonempty p_yes probabilities in [0, 1] and binary labels")
    return probabilities, labels.astype(np.int64)


def auroc(p: ArrayLike, y: ArrayLike) -> float | None:
    """Binary regression AUROC on p_yes; return None when only one class is observed."""
    probabilities, labels = _binary(p, y)
    return float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None


def ece_equal_mass(p: ArrayLike, y: ArrayLike, bins: int = 15) -> float:
    """Binary ECE after stable probability sorting and equal-count bins, including tied scores."""
    probabilities, labels = _binary(p, y)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    groups = np.array_split(np.argsort(probabilities, kind="stable"), min(bins, len(labels)))
    return float(sum(len(group) / len(labels) * abs(probabilities[group].mean() - labels[group].mean()) for group in groups))
