"""depth_to_3d 모듈 Unit Test (pytest / unittest 겸용).

plain assert 로 작성해 `pytest` 로도, `python -m pytest` 없이 아래처럼
직접 호출로도 돌릴 수 있다. 실행:
    pytest -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import depth_to_3d as d3  # noqa: E402


# --- fixtures ---------------------------------------------------------------
@pytest.fixture
def bgr_image():
    """4x4 BGR 테스트 이미지 (값이 알려진 결정론적 픽셀)."""
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, :, 2] = 100  # R 채널
    img[:, :, 1] = 150  # G 채널
    img[:, :, 0] = 200  # B 채널
    return img


# --- to_grayscale -----------------------------------------------------------
def test_to_grayscale_shape_and_dtype(bgr_image):
    gray = d3.to_grayscale(bgr_image)
    assert gray.shape == (4, 4)          # 채널 축 제거
    assert gray.dtype == np.uint8


def test_to_grayscale_known_value(bgr_image):
    # BT.601: 0.299*100 + 0.587*150 + 0.114*200 = 29.9+88.05+22.8 = 140.75 -> 141
    gray = d3.to_grayscale(bgr_image)
    assert np.all(gray == gray[0, 0])    # 균일 색이므로 전 픽셀 동일
    assert abs(int(gray[0, 0]) - 141) <= 1


def test_to_grayscale_passthrough_when_already_gray():
    g = np.arange(9, dtype=np.uint8).reshape(3, 3)
    out = d3.to_grayscale(g)
    assert np.array_equal(out, g)


def test_to_grayscale_none_raises():
    with pytest.raises(ValueError):
        d3.to_grayscale(None)


# --- normalize01 ------------------------------------------------------------
def test_normalize01_range():
    arr = np.array([[0, 128], [200, 255]], dtype=np.uint8)
    n = d3.normalize01(arr)
    assert n.dtype == np.float32
    assert n.min() == 0.0 and n.max() == 1.0


def test_normalize01_uniform_is_zero():
    arr = np.full((3, 3), 77, dtype=np.uint8)
    n = d3.normalize01(arr)
    assert np.all(n == 0.0)              # 0 나눗셈 방지 분기


# --- estimate_depth ---------------------------------------------------------
def test_estimate_depth_range():
    gray = (np.random.default_rng(0).integers(0, 256, (32, 32))).astype(np.uint8)
    depth = d3.estimate_depth(gray, blur_ksize=5)
    assert depth.shape == (32, 32)
    assert depth.dtype == np.float32
    assert 0.0 <= depth.min() and depth.max() <= 1.0


def test_estimate_depth_invert_flips_extremes():
    gray = np.tile(np.linspace(0, 255, 16, dtype=np.uint8), (16, 1))
    normal = d3.estimate_depth(gray, invert=False, blur_ksize=1)
    inverted = d3.estimate_depth(gray, invert=True, blur_ksize=1)
    # 정상: 밝은 쪽(우측)이 높음 / 반전: 어두운 쪽(좌측)이 높음
    assert normal[0, -1] > normal[0, 0]
    assert inverted[0, 0] > inverted[0, -1]
    assert np.allclose(normal + inverted, 1.0, atol=1e-5)


# --- depth_to_pointcloud ----------------------------------------------------
def test_pointcloud_count_full():
    depth = np.zeros((10, 8), dtype=np.float32)
    pts = d3.depth_to_pointcloud(depth, step=1)
    assert pts.shape == (80, 3)          # H*W 점
    assert pts.dtype == np.float32


def test_pointcloud_count_subsampled():
    depth = np.zeros((10, 8), dtype=np.float32)
    pts = d3.depth_to_pointcloud(depth, step=2)
    assert pts.shape == (5 * 4, 3)       # (10//2)*(8//2)


def test_pointcloud_z_equals_depth_times_scale():
    depth = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    pts = d3.depth_to_pointcloud(depth, step=1, z_scale=100.0)
    z_by_xy = {(int(p[0]), int(p[1])): p[2] for p in pts}
    assert z_by_xy[(0, 0)] == pytest.approx(0.0)
    assert z_by_xy[(1, 0)] == pytest.approx(50.0)   # col=1,row=0 -> depth 0.5
    assert z_by_xy[(0, 1)] == pytest.approx(100.0)  # col=0,row=1 -> depth 1.0


def test_pointcloud_bad_step_raises():
    with pytest.raises(ValueError):
        d3.depth_to_pointcloud(np.zeros((4, 4), np.float32), step=0)


# --- colors -----------------------------------------------------------------
def test_pointcloud_colors_align(bgr_image):
    colors = d3.pointcloud_colors(bgr_image, step=1)
    assert colors.shape == (16, 3)
    # BGR(200,150,100) -> RGB(100,150,200)
    assert tuple(colors[0]) == (100, 150, 200)


# --- save_ply ---------------------------------------------------------------
def test_save_ply_header_and_count(tmp_path):
    pts = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32)
    cols = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    path = tmp_path / "cloud.ply"
    d3.save_ply(str(path), pts, cols)
    text = path.read_text(encoding="ascii").splitlines()
    assert text[0] == "ply"
    assert "element vertex 2" in text
    # 헤더(end_header 까지) 뒤에 정확히 2개의 정점 줄
    body = text[text.index("end_header") + 1:]
    assert len([ln for ln in body if ln.strip()]) == 2


# --- backproject_pinhole ----------------------------------------------------
def test_backproject_pinhole_principal_point():
    depth = np.full((5, 5), 2.0, dtype=np.float32)
    pts = d3.backproject_pinhole(depth, fx=1.0, fy=1.0, cx=2.0, cy=2.0)
    # 주점(2,2)에서는 X=Y=0, Z=depth
    center = pts.reshape(5, 5, 3)[2, 2]
    assert center[0] == pytest.approx(0.0)
    assert center[1] == pytest.approx(0.0)
    assert center[2] == pytest.approx(2.0)


# --- generate_synthetic_scene ----------------------------------------------
def test_synthetic_scene_shape():
    scene = d3.generate_synthetic_scene(64)
    assert scene.shape == (64, 64)
    assert scene.dtype == np.uint8
    assert scene.max() > scene.min()     # 평평하지 않다


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
