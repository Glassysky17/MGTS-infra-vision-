"""Агент 2: предобработка."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from mgts.preprocessing.enhance import apply_clahe, auto_gamma, enhance
from mgts.preprocessing.quality import blur_score, brightness_stats, is_sharp


def _textured(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)


def test_blur_score_orders_sharp_over_blurred():
    sharp = _textured()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 6)
    assert blur_score(sharp) > 3 * blur_score(blurred)


def test_blur_score_black_image_is_zero():
    assert blur_score(np.zeros((100, 100, 3), np.uint8)) == 0.0


def test_is_sharp_threshold():
    sharp = _textured()
    assert is_sharp(sharp, 1.0)
    assert not is_sharp(cv2.GaussianBlur(sharp, (0, 0), 10), 1e6)


def test_brightness_stats_flags():
    dark = np.full((100, 100, 3), 10, np.uint8)
    bright = np.full((100, 100, 3), 250, np.uint8)
    assert brightness_stats(dark)["underexposed"] and not brightness_stats(dark)["overexposed"]
    assert brightness_stats(bright)["overexposed"]


def test_enhance_keeps_shape_dtype_and_input(cfg):
    img = _textured()
    before = img.copy()
    out = enhance(img, cfg)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(img, before), "enhance не должна мутировать вход"


def test_auto_gamma_brightens_dark_image():
    dark = np.full((120, 160, 3), 30, np.uint8)
    assert auto_gamma(dark).mean() > dark.mean() + 20


def test_clahe_increases_contrast_on_low_contrast():
    rng = np.random.default_rng(1)
    low = (110 + rng.integers(0, 12, (200, 200, 3))).astype(np.uint8)
    out = apply_clahe(low)
    assert out.std() > low.std()
