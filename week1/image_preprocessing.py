"""Food-101 이미지 전처리 파이프라인 (1차 업무).

Hugging Face `ethz/food101`에서 무작위로 5장을 받아 각 장마다 전처리를 모두 적용하고,
단계별 결과를 한 장의 비교 이미지로 저장한다.

실행:
    python image_preprocessing.py --num-samples 5 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset

from color_filter import apply_color_filter, color_pixel_ratio, detect_color_mask
from outlier import (
    DARK_MEAN_THRESHOLD,
    MIN_OBJECT_AREA_RATIO,
    inspect,
    object_mask,
)

TARGET_SIZE = (224, 224)

# ImageNet 사전학습 모델과 입력 분포를 맞추기 위한 통계값
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


# --------------------------------------------------------------------------
# 파이프라인
# --------------------------------------------------------------------------
def build_stages(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """원본 한 장에 전처리를 순서대로 적용해 (제목, 표시용 RGB) 목록을 만든다.

    matplotlib은 RGB를 기대하므로 표시 직전에 변환한다. 그림 안의 라벨은
    폰트 문제를 피하려고 영문으로 둔다.
    """
    base = resize(image)

    normalized = normalize(base)
    # 표준화 결과는 음수를 포함해 그대로 못 그리므로 되돌려 그린다.
    normalized_view = denormalize_for_display(normalized)

    blurred = denoise_blur(base)
    mask = detect_color_mask(base, "red")
    obj = object_mask(base)

    to_rgb = lambda bgr: cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray_to_rgb = lambda g: cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

    return [
        ("1. Original (224x224)", to_rgb(base)),
        ("2. Grayscale", gray_to_rgb(to_grayscale(base))),
        (
            f"3. Normalized (mean={normalized.mean():.2f}, std={normalized.std():.2f})",
            (normalized_view * 255).astype(np.uint8),
        ),
        ("4. Gaussian Blur 5x5", to_rgb(blurred)),
        ("5. Horizontal Flip", to_rgb(flip_horizontal(base))),
        ("6. Rotation +15deg", to_rgb(rotate(base))),
        ("7. Color Jitter (HSV)", to_rgb(color_jitter(base))),
        (f"8. Red Mask ({color_pixel_ratio(mask):.1%})", gray_to_rgb(mask)),
        ("9. Red Pixels Only", to_rgb(apply_color_filter(base, mask))),
        ("10. Object Mask (Otsu+CC)", gray_to_rgb(obj)),
    ]


def save_comparison(
    label: str, image: np.ndarray, index: int, output_dir: Path
) -> tuple[Path, object]:
    """한 장의 전 단계 결과를 격자 이미지로 저장하고 품질 판정을 함께 돌려준다."""
    base = resize(image)
    report = inspect(base)
    stages = build_stages(image)

    # axis("off")를 쓰면 tight_layout이 제목 높이를 제대로 못 잡아 아래 행 제목이
    # 위 행 이미지에 겹친다. constrained_layout은 suptitle까지 감안해 배치한다.
    fig, axes = plt.subplots(2, 5, figsize=(20, 9.2), constrained_layout=True)
    for ax, (title, panel) in zip(axes.ravel(), stages):
        ax.imshow(panel)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    verdict = "PASS" if report.passed else "REMOVE"
    fig.suptitle(
        f"[{index}] {label}  |  mean brightness {report.mean_brightness:.1f} "
        f"(>= {DARK_MEAN_THRESHOLD:.0f})  |  object area {report.object_area_ratio:.1%} "
        f"(>= {MIN_OBJECT_AREA_RATIO:.0%})  |  {verdict}",
        fontsize=13,
        fontweight="bold",
        color="#1a7f37" if report.passed else "#cf222e",
    )

    path = output_dir / f"{index}_{label}_steps.png"
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Food-101 전처리 파이프라인")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "preprocessed_samples")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.num_samples, args.seed)

    print(f"{'#':<3}{'label':<22}{'brightness':>11}{'obj area':>10}  verdict")
    print("-" * 62)
    kept = 0
    for i, (label, image) in enumerate(samples, start=1):
        path, report = save_comparison(label, image, i, args.output_dir)
        kept += report.passed
        print(
            f"{i:<3}{label:<22}{report.mean_brightness:>11.1f}"
            f"{report.object_area_ratio:>9.1%}  {report.reason}"
        )
    print("-" * 62)
    print(f"{kept}/{len(samples)} 장 통과, 결과 저장: {args.output_dir}")


if __name__ == "__main__":
    main()
