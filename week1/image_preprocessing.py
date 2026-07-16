"""Food-101 이미지 전처리 파이프라인 (1차 업무).

Hugging Face `ethz/food101`에서 무작위로 5장을 받아 각 장마다 전처리를 모두 적용하고,
단계별 결과를 한 장의 비교 이미지로 저장한다.

실행:
    python image_preprocessing.py --num-samples 5 --seed 42
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset

TARGET_SIZE = (224, 224)

# ImageNet 사전학습 모델과 입력 분포를 맞추기 위한 통계값
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 이상치 판정 임계값
DARK_MEAN_THRESHOLD = 40.0  # 평균 밝기(0~255)가 이보다 낮으면 '너무 어두움'
MIN_OBJECT_AREA_RATIO = 0.10  # 최대 객체가 프레임의 10% 미만이면 '객체가 너무 작음'


# --------------------------------------------------------------------------
# 데이터 적재
# --------------------------------------------------------------------------
def load_samples(num_samples: int = 5, seed: int = 42) -> list[tuple[str, np.ndarray]]:
    """Food-101에서 무작위 num_samples장을 BGR ndarray로 가져온다.

    전체 5GB를 내려받지 않도록 streaming 모드로 필요한 만큼만 읽는다.
    """
    ds = load_dataset("ethz/food101", split="train", streaming=True)
    label_names = ds.features["label"].names
    ds = ds.shuffle(seed=seed, buffer_size=500)

    samples = []
    for example in ds.take(num_samples):
        rgb = np.array(example["image"].convert("RGB"))
        # datasets는 PIL(RGB)로 주지만 OpenCV 연산은 BGR 기준이라 변환해 둔다.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        samples.append((label_names[example["label"]], bgr))
    return samples


# --------------------------------------------------------------------------
# 기본 전처리
# --------------------------------------------------------------------------
def resize(image: np.ndarray, size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """모델 입력 크기(224x224)로 통일한다."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """BGR -> 8bit 단일 채널. 내부적으로 0.299R+0.587G+0.114B 가중 평균."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize(image: np.ndarray) -> np.ndarray:
    """0~255 uint8 BGR -> ImageNet 통계로 표준화한 float32 RGB (H, W, 3)."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (rgb - IMAGENET_MEAN) / IMAGENET_STD


def denormalize_for_display(normalized: np.ndarray) -> np.ndarray:
    """표준화 결과는 음수를 포함해 그대로 못 그리므로 0~1로 되돌려 시각화용으로 쓴다."""
    return np.clip(normalized * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)


def denoise_blur(image: np.ndarray, ksize: int = 5, sigma: float = 0) -> np.ndarray:
    """가우시안 블러로 센서 노이즈/압축 아티팩트를 완화한다."""
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


# --------------------------------------------------------------------------
# 데이터 증강
# --------------------------------------------------------------------------
def flip_horizontal(image: np.ndarray) -> np.ndarray:
    """좌우 반전. 음식은 좌우 대칭성이 의미를 바꾸지 않아 안전한 증강이다."""
    return cv2.flip(image, 1)


def rotate(image: np.ndarray, angle: float = 15.0) -> np.ndarray:
    """중심 기준 회전. 빈 모서리는 반사 패딩으로 채워 검은 테두리를 막는다."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )


def color_jitter(
    image: np.ndarray,
    hue_shift: int = 10,
    sat_scale: float = 1.3,
    val_scale: float = 1.1,
) -> np.ndarray:
    """HSV 공간에서 색조/채도/명도를 흔들어 조명·카메라 차이를 흉내낸다."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
    # OpenCV의 Hue는 0~179 범위라 넘어가면 감싸 돌려야 한다.
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Food-101 전처리 파이프라인")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = load_samples(args.num_samples, args.seed)
    for label, image in samples:
        print(f"{label}: {image.shape} -> {resize(image).shape}")


if __name__ == "__main__":
    main()
