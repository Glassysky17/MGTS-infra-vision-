"""Агент 2: предобработка."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from mgts.preprocessing.enhance import apply_clahe, auto_gamma, denoise, enhance
from mgts.preprocessing.quality import (
    blur_score,
    brightness_stats,
    calibrate_blur_threshold,
    is_sharp,
    select_frames,
)


def _textured(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)


def test_blur_score_orders_sharp_over_blurred():
    sharp = _textured()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 6)
    assert blur_score(sharp) > 3 * blur_score(blurred)


def test_blur_score_black_image_is_zero():
    assert blur_score(np.zeros((100, 100, 3), np.uint8)) == 0.0


def test_blur_score_empty_image_is_zero():
    assert blur_score(np.empty((0, 20, 3), np.uint8)) == 0.0


def test_is_sharp_threshold():
    sharp = _textured()
    assert is_sharp(sharp, 1.0)
    assert not is_sharp(cv2.GaussianBlur(sharp, (0, 0), 10), 1e6)


def test_brightness_stats_flags():
    dark = np.full((100, 100, 3), 10, np.uint8)
    bright = np.full((100, 100, 3), 250, np.uint8)
    assert brightness_stats(dark)["underexposed"] and not brightness_stats(dark)["overexposed"]
    assert brightness_stats(bright)["overexposed"]


def test_brightness_stats_reports_gray_percentiles():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:5] = 100
    stats = brightness_stats(img)
    assert stats["mean"] == pytest.approx(50)
    assert stats["p5"] == pytest.approx(0)
    assert stats["p95"] == pytest.approx(100)


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


def test_enhance_disabled_returns_equal_copy(cfg):
    img = _textured()
    out = enhance(img, {"preprocessing": {"enhance": {"clahe": False, "gamma": None, "denoise": False}}})
    assert np.array_equal(out, img) and out is not img


def test_enhance_numeric_gamma_and_denoise(monkeypatch):
    img = np.full((40, 60, 3), 20, np.uint8)
    called = []
    monkeypatch.setattr("mgts.preprocessing.enhance.denoise", lambda image, strength=5: called.append(strength) or image)
    cfg = {"preprocessing": {"enhance": {"clahe": False, "gamma": 0.5, "denoise": True, "denoise_strength": 7}}}
    out = enhance(img, cfg)
    assert out.mean() > img.mean() and called == [7]


def test_denoise_preserves_shape_dtype():
    img = _textured()
    out = denoise(img, 3)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_select_frames_skips_unreadable_and_keeps_sharpest_duplicate(tmp_path):
    blurred = cv2.GaussianBlur(_textured(4), (0, 0), 5)
    pixels = blurred.reshape(-1, 3).copy()
    np.random.default_rng(4).shuffle(pixels)
    sharper = pixels.reshape(blurred.shape)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    other = tmp_path / "other.png"
    cv2.imwrite(str(first), blurred)
    cv2.imwrite(str(second), sharper)
    cv2.imwrite(str(other), _textured(8))
    paths = [str(first), str(second), str(tmp_path / "missing.png"), str(other)]
    selected = select_frames(paths, 1)
    assert selected == [str(second), str(other)]


def test_select_frames_evenly_limits_kept_frames(tmp_path):
    paths = []
    for i in range(5):
        path = tmp_path / f"{i}.png"
        image = np.random.default_rng(i).integers(5 + i * 45, 45 + i * 45, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(path), image)
        paths.append(str(path))
    selected = select_frames(paths, 0, max_keep=3)
    assert len(selected) == 3
    assert selected == sorted(selected, key=paths.index)


def test_calibrate_blur_threshold_uses_percentile(tmp_path):
    paths = []
    for i in range(3):
        path = tmp_path / f"{i}.png"
        cv2.imwrite(str(path), _textured(i))
        paths.append(str(path))
    expected = np.percentile([blur_score(cv2.imread(p)) for p in paths], 20)
    assert calibrate_blur_threshold(paths, 20) == pytest.approx(expected)


def test_calibrate_blur_threshold_empty_paths_returns_zero():
    assert calibrate_blur_threshold([]) == 0.0
