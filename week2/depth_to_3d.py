"""2D 이미지 -> 3D 포인트클라우드 변환 (Depth-as-Height 방식).

멘토 예제(cv2.applyColorMap으로 '가짜 depth')는 색만 입힐 뿐 3D 형상 정보를
만들지 못한다. 여기서는 픽셀 밝기를 '높이(Z)'로 해석하는 height-field 방식으로
실제 3차원 좌표를 만들고, 그 좌표를 .ply 로 저장/시각화한다.

핵심 설계: 모든 함수를 '순수 함수'(전역 상태 X, 파일 I/O 분리)로 만들어
unittest/pytest 로 입력->출력을 결정론적으로 검증할 수 있게 한다.

    depth_to_3d 파이프라인
    ┌────────────┐   ┌─────────────┐   ┌──────────────┐   ┌───────────────┐
    │ 2D 이미지   │──▶│ 그레이스케일 │──▶│ Depth Map    │──▶│ 3D PointCloud │
    │ (BGR/Gray) │   │  + 평활화    │   │ [0,1] 정규화 │   │  (X, Y, Z)    │
    └────────────┘   └─────────────┘   └──────────────┘   └───────────────┘
"""
from __future__ import annotations

import os

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 0) 유니코드 경로 안전 I/O (Windows: cv2.imread/imwrite 는 한글 경로에서 실패)
# ---------------------------------------------------------------------------
def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """한글/유니코드 경로에서도 되는 이미지 로드."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """한글/유니코드 경로에서도 되는 이미지 저장."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return bool(ok)


# ---------------------------------------------------------------------------
# 1) 픽셀 처리 (유닛테스트 대상)
# ---------------------------------------------------------------------------
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """BGR(3채널) 또는 이미 1채널인 이미지를 그레이스케일(CV_8U)로 변환.

    입력이 None 이면 ValueError (멘토 예제의 예외 처리 요구사항).
    """
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
    if image.ndim == 2:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"지원하지 않는 형태입니다: shape={image.shape}")


def normalize01(arr: np.ndarray) -> np.ndarray:
    """min-max 정규화. 결과는 float32, 값 범위 [0, 1].

    max == min 인 균일 이미지는 0 나눗셈을 피해 전부 0 으로 반환한다.
    """
    a = arr.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return (a - lo) / (hi - lo)


def estimate_depth(gray: np.ndarray, invert: bool = False, blur_ksize: int = 5) -> np.ndarray:
    """그레이스케일 -> Depth Map([0,1] float32).

    - blur_ksize>1 이면 가우시안 평활화로 센서 노이즈를 줄여 표면을 매끄럽게 한다.
    - invert=True 면 '밝을수록 멀다' 규약(어두운 곳이 가까움)으로 뒤집는다.
      기본은 '밝을수록 높다(가깝다)'.
    """
    if gray.ndim != 2:
        raise ValueError("estimate_depth 는 단일 채널 이미지를 받는다.")
    work = gray
    if blur_ksize and blur_ksize > 1:
        k = blur_ksize | 1  # 홀수 보정
        work = cv2.GaussianBlur(gray, (k, k), 0)
    depth = normalize01(work)
    if invert:
        depth = 1.0 - depth
    return depth


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Depth([0,1] 또는 임의 범위)를 COLORMAP_JET 컬러(BGR uint8)로 시각화."""
    d8 = (normalize01(depth) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_JET)


# ---------------------------------------------------------------------------
# 2) 2D -> 3D 변환
# ---------------------------------------------------------------------------
def depth_to_pointcloud(
    depth: np.ndarray, step: int = 1, z_scale: float = 100.0
) -> np.ndarray:
    """Depth Map -> 3D 포인트클라우드 (N, 3), 각 행이 (X, Y, Z).

    X = 열(col), Y = 행(row), Z = depth * z_scale.
    step 으로 격자를 솎아 점 개수를 (H//step)*(W//step) 로 줄일 수 있다.
    반환 dtype 은 float32.
    """
    if depth.ndim != 2:
        raise ValueError("depth 는 2차원이어야 한다.")
    if step < 1:
        raise ValueError("step 은 1 이상이어야 한다.")
    h, w = depth.shape
    rows = np.arange(0, h, step)
    cols = np.arange(0, w, step)
    xx, yy = np.meshgrid(cols, rows)          # xx: 열(X), yy: 행(Y)
    zz = depth[np.ix_(rows, cols)] * z_scale
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return points.astype(np.float32)


def pointcloud_colors(image: np.ndarray, step: int = 1) -> np.ndarray:
    """포인트클라우드 각 점에 대응하는 원본 색(RGB uint8, (N,3))을 뽑아낸다.

    depth_to_pointcloud 와 같은 step 격자 순서를 따르므로 점-색이 1:1 대응한다.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    rows = np.arange(0, h, step)
    cols = np.arange(0, w, step)
    sub = image[np.ix_(rows, cols)]                    # (r, c, 3) BGR
    rgb = sub[:, :, ::-1].reshape(-1, 3)               # BGR -> RGB
    return rgb.astype(np.uint8)


# --- 보너스: 실제 깊이(미터)를 카메라 좌표로 역투영하는 핀홀 방식 ---
def backproject_pinhole(
    depth: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """핀홀 카메라 역투영. 실제 depth(예: 스테레오 결과)를 3D 점으로.

    X = (c - cx) * d / fx,  Y = (r - cy) * d / fy,  Z = d
    """
    h, w = depth.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32)
    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy
    return np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# 3) 저장 (I/O 경계 — 순수 로직과 분리)
# ---------------------------------------------------------------------------
def save_ply(path: str, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """포인트클라우드를 표준 ASCII PLY 로 저장 (Open3D/MeshLab/CloudCompare 호환)."""
    n = points.shape[0]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if colors is not None:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header.append("end_header")

    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(header) + "\n")
        if colors is None:
            for p in points:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        else:
            for p, c in zip(points, colors):
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


# ---------------------------------------------------------------------------
# 4) 합성 장면 (결정론적 -> 테스트/데모용)
# ---------------------------------------------------------------------------
def generate_synthetic_scene(size: int = 200) -> np.ndarray:
    """알고리즘 검증용 합성 그레이스케일 장면.

    좌측엔 반구(hemisphere), 배경엔 대각 램프(ramp). 높이의 정답을 알고 있어
    3D 복원 결과가 맞는지 눈으로/수치로 확인하기 좋다. 반환: CV_8U (size, size).
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    ramp = (xx + yy) / (2 * (size - 1))                 # 0..1 대각 경사

    cx, cy, r = size * 0.35, size * 0.5, size * 0.28
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    inside = dist2 <= r * r
    hemi = np.zeros_like(ramp)
    hemi[inside] = np.sqrt(np.clip(r * r - dist2[inside], 0, None)) / r  # 0..1 반구

    scene = np.maximum(ramp * 0.6, hemi)                # 반구가 배경 위로 솟음
    return (scene * 255).astype(np.uint8)
