import numpy as np
import pytest

from vehicle_recognition.preprocessing.roi import clip_box, crop_roi, iou


def test_clip_and_crop_inside():
    img = np.zeros((100, 200, 3), np.uint8)
    assert crop_roi(img, (10, 20, 110, 70)).shape == (50, 100, 3)


def test_clip_out_of_bounds_and_swapped():
    assert clip_box((-5, -5, 300, 120), 200, 100) == (0, 0, 200, 100)
    assert clip_box((50, 60, 10, 20), 200, 100) == (10, 20, 50, 60)


def test_padding_clipped():
    img = np.zeros((100, 100, 3), np.uint8)
    assert crop_roi(img, (0, 0, 50, 50), pad_frac=0.5).shape == (75, 75, 3)


def test_degenerate_and_nan():
    img = np.zeros((100, 100, 3), np.uint8)
    with pytest.raises(ValueError):
        crop_roi(img, (150, 150, 200, 200))
    with pytest.raises(ValueError):
        crop_roi(img, (np.nan, 0, 10, 10))


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (10, 10, 20, 20)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 50 / 150) < 1e-9
