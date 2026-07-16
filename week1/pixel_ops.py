"""픽셀 단위 직접 구현 (OpenCV 대조군).

image_preprocessing.py는 실무 관행대로 OpenCV를 쓴다. 이 모듈은 같은 연산을
NumPy 인덱싱/컨볼루션으로 다시 구현해서, cv2 함수가 픽셀 수준에서 무엇을 하는지
드러내고 두 결과가 실제로 일치하는지 수치로 확인하기 위한 것이다.

실행하면 cv2 대비 오차 표를 출력한다:
    python pixel_ops.py
"""

from __future__ import annotations

import cv2
import numpy as np

# cv2.cvtColor(BGR2GRAY)가 내부적으로 쓰는 ITU-R BT.601 계수 (B, G, R 순서)
BT601_BGR = np.array([0.114, 0.587, 0.299], dtype=np.float32)


def grayscale_numpy(image: np.ndarray) -> np.ndarray:
    """BGR 각 픽셀에 BT.601 가중치를 곱해 더한다.

    cv2는 정수 고정소수점으로 계산하고 반올림하므로, float 연산 후 반올림해야
    결과가 맞아떨어진다.
    """
    weighted = image.astype(np.float32) @ BT601_BGR
    return np.clip(np.rint(weighted), 0, 255).astype(np.uint8)


# OpenCV는 sigma를 안 주고 커널이 7 이하로 작으면, 가우시안 공식 대신 아래의
# 하드코딩된 이항계수 테이블을 쓴다 (OpenCV smallGaussianTab). 공식을 그대로 쓰면
# cv2와 결과가 눈에 띄게 갈라지므로 같은 분기를 재현한다.
SMALL_GAUSSIAN_TAB = {
    1: [1.0],
    3: [0.25, 0.5, 0.25],
    5: [0.0625, 0.25, 0.375, 0.25, 0.0625],
    7: [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
}


def gaussian_kernel_1d(ksize: int, sigma: float) -> np.ndarray:
    """1차원 가우시안 커널. sigma<=0이면 cv2와 같은 규칙으로 유도한다."""
    if sigma <= 0 and ksize in SMALL_GAUSSIAN_TAB:
        return np.array(SMALL_GAUSSIAN_TAB[ksize], dtype=np.float32)
    if sigma <= 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    radius = ksize // 2
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur_numpy(image: np.ndarray, ksize: int = 5, sigma: float = 0) -> np.ndarray:
    """2D 가우시안을 수평/수직 1D 컨볼루션 두 번으로 분리해 적용한다.

    분리 가능(separable) 성질 덕에 픽셀당 곱셈이 k^2회에서 2k회로 줄어든다.
    가장자리는 cv2.GaussianBlur의 기본값과 같은 reflect_101로 채운다.
    """
    kernel = gaussian_kernel_1d(ksize, sigma)
    radius = ksize // 2
    work = image.astype(np.float32)
    if work.ndim == 2:
        work = work[:, :, None]

    padded = np.pad(work, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")

    # 수평 방향: 열을 radius만큼 밀어가며 가중합
    horizontal = np.zeros_like(padded)
    for i, weight in enumerate(kernel):
        horizontal += weight * np.roll(padded, radius - i, axis=1)

    # 수직 방향
    blurred = np.zeros_like(horizontal)
    for i, weight in enumerate(kernel):
        blurred += weight * np.roll(horizontal, radius - i, axis=0)

    blurred = blurred[radius:-radius, radius:-radius]
    result = np.clip(np.rint(blurred), 0, 255).astype(np.uint8)
    return result[:, :, 0] if image.ndim == 2 else result


def flip_horizontal_numpy(image: np.ndarray) -> np.ndarray:
    """열 인덱스를 뒤집는다. 픽셀 값 자체는 손대지 않는 순수 재배치."""
    return image[:, ::-1].copy()


def rotate_numpy(image: np.ndarray, angle: float = 15.0) -> np.ndarray:
    """역방향 매핑 + 이중선형 보간으로 회전한다.

    출력 픽셀에서 원본 좌표를 거꾸로 찾아가는(inverse mapping) 방식이라
    구멍이 생기지 않는다. 원본 범위를 벗어난 좌표는 가장자리 값으로 고정한다.
    """
    h, w = image.shape[:2]
    theta = np.deg2rad(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cx, cy = w / 2.0, h / 2.0

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    # 순방향은 dst = R(dst-c)+c, R = [[cos, sin], [-sin, cos]] (cv2와 동일 규약).
    # 여기선 출력에서 원본을 거꾸로 찾으므로 R의 역행렬 [[cos, -sin], [sin, cos]]을 쓴다.
    dx, dy = xx - cx, yy - cy
    src_x = cos_t * dx - sin_t * dy + cx
    src_y = sin_t * dx + cos_t * dy + cy

    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)

    x0, y0 = np.floor(src_x).astype(np.int32), np.floor(src_y).astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = (src_x - x0)[..., None], (src_y - y0)[..., None]

    src = image.astype(np.float32)
    if src.ndim == 2:
        src = src[:, :, None]

    # 이웃 4픽셀을 x축으로 섞고, 그 둘을 다시 y축으로 섞는다
    top = src[y0, x0] * (1 - wx) + src[y0, x1] * wx
    bottom = src[y1, x0] * (1 - wx) + src[y1, x1] * wx
    out = np.clip(np.rint(top * (1 - wy) + bottom * wy), 0, 255).astype(np.uint8)
    return out[:, :, 0] if image.ndim == 2 else out


def compare(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """두 구현 결과의 평균 절대오차와 최대 절대오차(0~255 스케일)."""
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return float(diff.mean()), float(diff.max())


def _cv_rotate_replicate(image: np.ndarray, angle: float) -> np.ndarray:
    """비교 전용 회전. rotate_numpy가 범위 밖 좌표를 가장자리 값으로 고정하므로,
    cv2 쪽도 BORDER_REPLICATE로 맞춰야 경계 처리 차이가 오차로 잡히지 않는다.
    (image_preprocessing.rotate가 쓰는 반사 패딩과는 다른 설정이다.)
    """
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def main() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)

    rows = [
        ("grayscale", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), grayscale_numpy(image)),
        (
            "gaussian blur 5x5",
            cv2.GaussianBlur(image, (5, 5), 0),
            gaussian_blur_numpy(image, 5, 0),
        ),
        ("flip horizontal", cv2.flip(image, 1), flip_horizontal_numpy(image)),
        ("rotate 15deg", _cv_rotate_replicate(image, 15.0), rotate_numpy(image, 15.0)),
    ]

    print(f"{'operation':<20}{'mean abs err':>14}{'max abs err':>13}")
    for name, cv_out, np_out in rows:
        mean_err, max_err = compare(cv_out, np_out)
        print(f"{name:<20}{mean_err:>14.4f}{max_err:>13.1f}")


if __name__ == "__main__":
    main()
