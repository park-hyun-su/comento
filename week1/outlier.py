"""이상치 탐지 및 필터링 (심화 문제).

학습 데이터에 섞이면 해로운 두 종류를 걸러낸다.
  1. 너무 어두운 이미지 — 평균 밝기 기준
  2. 객체가 너무 작은 이미지 — 최대 연결요소의 면적 비율 기준
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DARK_MEAN_THRESHOLD = 40.0
MIN_OBJECT_AREA_RATIO = 0.10


@dataclass
class QualityReport:
    """한 장에 대한 이상치 판정 결과."""

    mean_brightness: float
    object_area_ratio: float
    too_dark: bool
    object_too_small: bool

    @property
    def passed(self) -> bool:
        return not (self.too_dark or self.object_too_small)

    @property
    def reason(self) -> str:
        reasons = []
        if self.too_dark:
            reasons.append(f"어두움(평균 {self.mean_brightness:.1f})")
        if self.object_too_small:
            reasons.append(f"객체 작음({self.object_area_ratio:.1%})")
        return ", ".join(reasons) if reasons else "PASS"


def mean_brightness(image: np.ndarray) -> float:
    """그레이스케일 평균 밝기(0~255)."""
    return float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())


def is_too_dark(image: np.ndarray, threshold: float = DARK_MEAN_THRESHOLD) -> bool:
    return mean_brightness(image) < threshold


def _border_touch_ratio(mask: np.ndarray) -> float:
    """마스크가 이미지 테두리를 얼마나 차지하는지(0~1).

    배경은 보통 프레임 가장자리를 따라 이어지고 피사체는 그렇지 않다.
    전경/배경을 가려낼 때 이 값을 근거로 쓴다.
    """
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return float(np.count_nonzero(border)) / border.size


def _foreground_mask(image: np.ndarray) -> np.ndarray:
    """Otsu 이진화로 전경 후보를 만든다.

    Otsu는 밝기 기준으로 두 덩어리를 나눠줄 뿐, 어느 쪽이 피사체인지는 알려주지
    않는다. 음식 사진은 접시가 밝은 경우와 어두운 경우가 섞여 있어 한쪽으로
    고정할 수 없다. 그래서 마스크와 그 반전을 모두 후보로 두고, 테두리를 덜
    차지하는 쪽을 전경으로 택한다.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    candidates = []
    for mask in (otsu, cv2.bitwise_not(otsu)):
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        candidates.append(cleaned)

    return min(candidates, key=_border_touch_ratio)


def object_mask(image: np.ndarray) -> np.ndarray:
    """전경에서 가장 큰 연결요소만 남긴 마스크."""
    foreground = _foreground_mask(image)
    # 라벨 0은 배경이므로 제외하고 가장 넓은 연결요소를 고른다.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    if count <= 1:
        return np.zeros(image.shape[:2], np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def object_area_ratio(image: np.ndarray) -> float:
    """가장 큰 객체가 프레임에서 차지하는 면적 비율(0~1)."""
    return float(np.count_nonzero(object_mask(image))) / (image.shape[0] * image.shape[1])


def inspect(image: np.ndarray) -> QualityReport:
    """이미지 한 장의 이상치 지표를 계산한다."""
    brightness = mean_brightness(image)
    ratio = object_area_ratio(image)
    return QualityReport(
        mean_brightness=brightness,
        object_area_ratio=ratio,
        too_dark=brightness < DARK_MEAN_THRESHOLD,
        object_too_small=ratio < MIN_OBJECT_AREA_RATIO,
    )
