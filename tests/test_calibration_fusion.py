import numpy as np
import pytest
from scipy.special import softmax

from vehicle_recognition.calibration.calibration import apply_temperature, brier, ece, fit_temperature, nll
from vehicle_recognition.calibration.selective import decide, select_threshold
from vehicle_recognition.evaluation.metrics import check_probs, classification_report, topk_accuracy
from vehicle_recognition.fusion.fusion import FusionUnavailableError, fuse, select_alpha


def test_ece_perfect_and_overconfident():
    y = np.array([0, 1, 0, 1])
    p = np.eye(2)[y]
    assert ece(p, y) == 0.0
    p_bad = np.array([[0.99, 0.01]] * 4)
    assert abs(ece(p_bad, y) - 0.49) < 1e-9


def test_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 2, (4000, 10))
    y = np.array([rng.choice(10, p=softmax(row)) for row in true_logits])
    t = fit_temperature(true_logits * 3.0, y)
    assert 2.6 < t < 3.4
    p = apply_temperature(true_logits * 3.0, t)
    assert ece(p, y) < ece(softmax(true_logits * 3.0, axis=1), y)


def test_nll_brier_bounds():
    y = np.array([0, 1])
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert nll(p, y) < 1e-9 and brier(p, y) == 0.0


def test_fusion_and_alpha():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 300)
    good = softmax(np.eye(5)[y] * 3 + rng.normal(0, 1, (300, 5)), axis=1)
    noise = softmax(rng.normal(0, 1, (300, 5)), axis=1)
    sel = select_alpha(good, noise, y)
    assert sel["alpha"] >= 0.8
    f = fuse(good, noise, 0.5)
    check_probs(f, 5)
    with pytest.raises(FusionUnavailableError):
        fuse(good, None, 0.5)
    with pytest.raises(ValueError):
        fuse(good, noise[:, :4], 0.5)


def test_check_probs_rejects_bad():
    with pytest.raises(ValueError):
        check_probs(np.array([[0.5, 0.6]]))
    with pytest.raises(ValueError):
        check_probs(np.array([[np.nan, 1.0]]))


def test_topk_and_report():
    p = np.array([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]])
    y = np.array([1, 0])
    assert topk_accuracy(p, y, 1) == 0.5 and topk_accuracy(p, y, 2) == 1.0
    r = classification_report(p, y, 3)
    assert r["top1"] == 0.5


def test_threshold_and_decide():
    p = np.array([[0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.55, 0.45]])
    y = np.array([0, 0, 1, 0])
    s = select_threshold(p, y, target_selective_accuracy=1.0)
    assert s["threshold"] == 0.8 and s["coverage"] == 0.5
    assert decide(p[2], s["threshold"])["rejected"] is True
    assert decide(p[0], s["threshold"])["rejected"] is False
    assert decide(p[2], None)["rejected"] is False
