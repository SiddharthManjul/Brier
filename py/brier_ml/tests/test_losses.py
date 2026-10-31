from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from brier_ml.losses import (
    accuracy,
    auroc,
    brier,
    ece_equal_mass,
    nll,
    option_probs,
    proper_loss,
    rps,
)


def _batch() -> tuple[torch.Tensor, ...]:
    probabilities = torch.tensor([[.25, .75, 1, 1, 1, 1], [1 / 6] * 6,
                                  [.1, .2, .3, .15, .25, 1]], dtype=torch.float64)
    logits = probabilities.log().requires_grad_()
    valid = torch.arange(6)[None, :] < torch.tensor([2, 6, 5])[:, None]
    labels = torch.tensor([1, 3, 1])
    score = torch.tensor([False, False, True])
    ordinal = torch.tensor([[-1] * 6, [-1] * 6, [4, 1, 0, 3, 2, -1]])
    return logits, valid, labels, score, ordinal


def test_option_probs_masks_and_temperature() -> None:
    logits, valid, *_ = _batch()
    p = option_probs(logits, valid)
    torch.testing.assert_close(p.sum(-1), torch.ones(3, dtype=p.dtype))
    assert p[~valid].eq(0).all()
    torch.testing.assert_close(p[0, :2], torch.tensor([.25, .75], dtype=p.dtype))
    torch.testing.assert_close(option_probs(logits, valid, 2)[0, :2],
                               torch.tensor([1 / (1 + math.sqrt(3)), math.sqrt(3) / (1 + math.sqrt(3))], dtype=p.dtype))


def test_proper_loss_mixed_questions_hand_computed_and_gradients() -> None:
    args = _batch()
    loss, parts = proper_loss(*args)
    ce = (-math.log(.75) + math.log(6) - math.log(.2)) / 3
    extra = (.125 + 5 / 6 + .103125) / 3
    assert parts["ce"].item() == pytest.approx(ce)
    assert parts["extra"].item() == pytest.approx(extra)
    assert loss.item() == pytest.approx(ce + extra)
    loss.backward()
    assert torch.isfinite(args[0].grad).all()
    assert args[0].grad[~args[1]].eq(0).all()
    assert args[0].grad[args[1]].abs().sum() > 0
    assert proper_loss(*args, lam=0)[0].item() == pytest.approx(ce)
    assert proper_loss(*args, lam=2)[0].item() == pytest.approx(ce + 2 * extra)


def test_stable_ce_retains_gradient_for_underflowed_target() -> None:
    logits = torch.tensor([[1000., -1000., 0., 0., 0., 0.]], requires_grad=True)
    valid = torch.tensor([[True, True, False, False, False, False]])
    loss, parts = proper_loss(logits, valid, torch.tensor([1]), torch.tensor([False]), torch.full((1, 6), -1))
    assert parts["ce"].item() == 2000
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[1., -1., 0., 0., 0., 0.]]))


def test_ordinal_loss_is_invariant_to_option_shuffle_and_noncontiguous_mask() -> None:
    args = _batch()
    permutation = torch.tensor([5, 4, 2, 0, 3, 1])
    shuffled = (args[0][:, permutation], args[1][:, permutation],
                torch.argsort(permutation)[args[2]], args[3], args[4][:, permutation])
    assert proper_loss(*args)[0].item() == pytest.approx(proper_loss(*shuffled)[0].item())


def test_half_precision_probabilities_and_loss_use_float32() -> None:
    args = _batch()
    logits = args[0].detach().to(torch.bfloat16).requires_grad_()
    assert option_probs(logits, args[1]).dtype == torch.float32
    loss, _ = proper_loss(logits, *args[1:])
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_probability_normalization_fails_clearly_on_float_overflow() -> None:
    logits = torch.tensor([[3e38, -3e38, 0, 0, 0, 0]])
    valid = torch.tensor([[True, True, False, False, False, False]])
    with pytest.raises(ValueError, match="floating-point range"):
        option_probs(logits, valid)


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf"), True, "1", 1e-310])
def test_option_probs_rejects_invalid_temperature(temperature: object) -> None:
    with pytest.raises(ValueError, match="temperature"):
        option_probs(*_batch()[:2], temperature=temperature)


@pytest.mark.parametrize("change", ["shape", "empty", "integer", "nan", "mask_shape", "mask_type", "zero", "three"])
def test_option_probs_rejects_invalid_logits_or_mask(change: str) -> None:
    logits, valid, *_ = _batch()
    if change == "shape":
        logits = logits[:, :5]
    elif change == "empty":
        logits, valid = logits[:0], valid[:0]
    elif change == "integer":
        logits = logits.long()
    elif change == "nan":
        logits = logits.detach().clone()
        logits[0, 5] = torch.nan
    elif change == "mask_shape":
        valid = valid[:1]
    elif change == "mask_type":
        valid = valid.long()
    else:
        valid = valid.clone()
        valid[0] = torch.arange(6) < (0 if change == "zero" else 3)
    with pytest.raises(ValueError, match="logits"):
        option_probs(logits, valid)


@pytest.mark.parametrize("change", ["label_shape", "label_type", "negative", "overflow", "invalid_label", "score_type",
                                    "score_shape", "map_type", "map_shape", "duplicate", "unused", "nonscore", "score_k"])
def test_proper_loss_rejects_invalid_targets_and_maps(change: str) -> None:
    logits, valid, labels, score, ordinal = _batch()
    if change == "label_shape":
        labels = labels[:, None]
    elif change == "label_type":
        labels = labels.float()
    elif change in {"negative", "overflow", "invalid_label"}:
        labels[0] = {"negative": -1, "overflow": 6, "invalid_label": 2}[change]
    elif change == "score_type":
        score = score.long()
    elif change == "score_shape":
        score = score[:1]
    elif change == "map_type":
        ordinal = ordinal.float()
    elif change == "map_shape":
        ordinal = ordinal[:, :5]
    elif change == "duplicate":
        ordinal[2, 0] = 1
    elif change == "unused":
        ordinal[2, 5] = 5
    elif change == "nonscore":
        ordinal[0, 0] = 0
    else:
        score[0] = True
    with pytest.raises(ValueError):
        proper_loss(logits, valid, labels, score, ordinal)


@pytest.mark.parametrize("lam", [-1, float("nan"), float("inf"), True, "1"])
def test_proper_loss_rejects_invalid_weight(lam: object) -> None:
    with pytest.raises(ValueError, match="lam"):
        proper_loss(*_batch(), lam=lam)


def test_nll_hand_computed() -> None:
    assert nll([[.25, .75], [.8, .2]], [1, 0]) == pytest.approx(-(math.log(.75) + math.log(.8)) / 2)
    assert nll([[1, 0]], [1]) == pytest.approx(-math.log(1e-12))


def test_brier_is_multiclass_sum_of_squares() -> None:
    assert brier([[.25, .75], [.8, .2]], [1, 0]) == pytest.approx((.125 + .08) / 2)
    assert brier([[1 / 6] * 6], [3]) == pytest.approx(5 / 6)


def test_rps_hand_computed_ordinal_probabilities() -> None:
    assert rps([[.3, .2, .25, .15, .1]], [1]) == pytest.approx(.103125)
    assert rps([[1, 0, 0, 0, 0]], [4]) == 1
    assert rps([[0, 0, 0, 0, 1]], [4]) == 0


def test_accuracy_and_tie_breaking() -> None:
    assert accuracy([[.2, .8], [.5, .5], [.7, .3]], [1, 1, 0]) == pytest.approx(2 / 3)


def test_auroc_on_p_yes_and_single_class() -> None:
    assert auroc([.1, .4, .35, .8], [0, 0, 1, 1]) == .75
    assert auroc([.2, .2], [0, 1]) == .5
    assert auroc([.1, .2], [0, 0]) is None


def test_ece_equal_mass_hand_computed_sorted_bins_and_ties() -> None:
    assert ece_equal_mass([.9, .2, .8, .1], [1, 1, 0, 0], bins=2) == pytest.approx(.35)
    assert ece_equal_mass([.1, .2, .4, .6, .9], [0, 0, 1, 1, 1], bins=2) == pytest.approx(.16)
    assert ece_equal_mass([.5] * 30, [0] * 15 + [1] * 15) == pytest.approx(14 / 30)
    assert ece_equal_mass([.2, .8], [0, 1]) == pytest.approx(.2)


@pytest.mark.parametrize("metric", [nll, brier, rps, accuracy])
@pytest.mark.parametrize("p,y", [([], []), ([[.2, .3]], [0]), ([[.2, .8]], [2]), ([[.2, .8]], [-1]),
                                ([[.2, .8]], [1.0]), ([[.2, .8]], [[1]]), ([[np.nan, 0]], [0]),
                                ([[-.1, 1.1]], [0]), ([[.2, .8], [.4, .6]], [0])])
def test_multiclass_metrics_reject_bad_inputs(metric: object, p: object, y: object) -> None:
    with pytest.raises(ValueError):
        metric(p, y)


@pytest.mark.parametrize("metric", [auroc, ece_equal_mass])
@pytest.mark.parametrize("p,y", [([], []), ([.5], []), ([[.5]], [1]), ([np.inf], [1]), ([-.1], [0]),
                                ([1.1], [1]), ([.5], [2]), ([.5], [np.nan])])
def test_binary_metrics_reject_bad_inputs(metric: object, p: object, y: object) -> None:
    with pytest.raises(ValueError):
        metric(p, y)


@pytest.mark.parametrize("bins", [0, -1, 1.5, True, "15"])
def test_ece_rejects_invalid_bin_count(bins: object) -> None:
    with pytest.raises(ValueError, match="bins"):
        ece_equal_mass([.2, .8], [0, 1], bins=bins)
