"""특정 색상 픽셀 감지 및 필터링.

업무 요청서 [요청내용] 2의 "특정 색상의 픽셀을 감지하고 필터링" 항목.

RGB에서 "빨간색"을 직접 고르려면 밝기에 따라 값이 크게 흔들려서 조건이 지저분해진다.
HSV는 색상(H)이 밝기(V)와 분리돼 있어, 조명이 달라져도 같은 H 범위로 잡힌다.
음식 사진에서는 토마토/고기/소스 계열을 골라내는 데 쓴다.
"""

from __future__ import annotations

import cv2
import numpy as np

# OpenCV HSV 범위: H 0~179, S 0~255, V 0~255.
# 빨강은 H=0을 기준으로 양쪽으로 갈라져 있어 두 구간을 따로 잡아 합쳐야 한다.
COLOR_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, 90, 60), (10, 255, 255)), ((170, 90, 60), (179, 255, 255))],
    "yellow": [((20, 90, 60), (35, 255, 255))],
    "green": [((36, 60, 40), (85, 255, 255))],
}


def detect_color_mask(image: np.ndarray, color: str = "red") -> np.ndarray:
    """지정 색상에 해당하는 픽셀은 255, 나머지는 0인 8bit 마스크를 만든다.

    S/V 하한을 둬서 회색빛 접시나 어두운 그림자가 색상으로 오인되는 걸 막는다.
    """
    if color not in COLOR_RANGES:
        raise ValueError(f"지원하지 않는 색상: {color} (가능: {list(COLOR_RANGES)})")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for lower, upper in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))

    # 점처럼 흩어진 오검출은 열기 연산으로 없애고, 객체 내부의 구멍은 닫기로 메운다.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def apply_color_filter(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """마스크가 켜진 픽셀만 남기고 나머지는 검게 만든다."""
    return cv2.bitwise_and(image, image, mask=mask)


def color_pixel_ratio(mask: np.ndarray) -> float:
    """마스크가 차지하는 픽셀 비율(0~1)."""
    return float(np.count_nonzero(mask)) / mask.size


def describe_pixel(image: np.ndarray, x: int, y: int) -> str:
    """특정 좌표 한 픽셀의 BGR/HSV 값을 문자열로. 픽셀 단위 분석 확인용."""
    b, g, r = image[y, x]
    h, s, v = cv2.cvtColor(image[y : y + 1, x : x + 1], cv2.COLOR_BGR2HSV)[0, 0]
    return f"({x},{y}) BGR=({b},{g},{r}) HSV=({h},{s},{v})"
