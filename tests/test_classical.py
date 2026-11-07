import numpy as np

from vehicle_recognition.classical.features import FeatureConfig, extract_features, feature_dims
from vehicle_recognition.classical.model import (
    ClassicalSpec,
    build_pipeline,
    predict_proba_canonical,
    stack_families,
)


def test_feature_dims_and_finiteness():
    cfg = FeatureConfig()
    dims = feature_dims(cfg)
    # HOG: 160x96 / 16px cells -> 10x6 cells -> 9x5 blocks * 2*2*9
    assert dims["hog"] == 9 * 5 * 2 * 2 * 9
    assert dims["lbp"] == 4 * 4 * (8 + 2)
    assert dims["hsv"] == 8 * 4 * 4
    rng = np.random.default_rng(0)
    f = extract_features(rng.integers(0, 255, (77, 233, 3), dtype=np.uint8), cfg)
    assert {k: v.shape[0] for k, v in f.items()} == dims
    assert np.isclose(f["hsv"].sum(), 1.0, atol=1e-5)
    assert np.isclose(f["lbp"].sum(), 16.0, atol=1e-4)  # 16 normalised cell histograms


def test_svm_canonical_class_order_with_missing_classes():
    rng = np.random.default_rng(0)
    present = [3, 7, 50, 195]
    X = np.vstack([rng.normal(i, 0.3, (12, 6)) for i in range(len(present))]).astype(np.float32)
    y = np.repeat(present, 12)
    feats = {"a": X[:, :4], "b": X[:, 4:]}
    Xs, dims = stack_families(feats, ("a", "b"))
    pipe = build_pipeline(ClassicalSpec(pca_components=None, C=1.0), dims).fit(Xs, y)
    p = predict_proba_canonical(pipe, Xs, 196)
    assert p.shape == (48, 196)
    assert np.allclose(p.sum(1), 1)
    assert set(np.nonzero(p.sum(0))[0]) == set(present)
    assert np.mean(p.argmax(1) == y) > 0.9
