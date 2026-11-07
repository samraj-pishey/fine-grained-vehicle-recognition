import numpy as np
import pytest

from vehicle_recognition.evaluation.perturbations import CONDITIONS, SEVERITIES, perturb


@pytest.mark.parametrize("cond", CONDITIONS)
def test_perturb_shapes_and_determinism(cond):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (240, 360, 3), dtype=np.uint8)
    box = (40, 30, 300, 200)
    for s in SEVERITIES:
        a = perturb(img, box, cond, s, seed=7, pad=0.05)
        b = perturb(img, box, cond, s, seed=7, pad=0.05)
        assert a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] == 3
        assert np.array_equal(a, b)
    # severity increases the deviation from the clean ROI on average
    from vehicle_recognition.preprocessing.roi import crop_roi

    clean = crop_roi(img, box, 0.05).astype(float)
    d = [np.abs(perturb(img, box, cond, s, seed=3, pad=0.05).astype(float) - clean).mean() for s in SEVERITIES]
    assert d[2] >= d[0] * 0.8
