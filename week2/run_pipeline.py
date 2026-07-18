"""엔드투엔드 실행: 2D 이미지 -> Depth Map -> 3D 포인트클라우드 -> 저장/시각화.

사용법:
    python run_pipeline.py                 # 합성 장면으로 데모
    python run_pipeline.py path/to/img.jpg # 실제 이미지로 실행

결과는 outputs/ 에 저장된다:
    <name>_panels.png   4분할 결과 이미지(원본/Depth/3D표면/포인트클라우드)
    <name>_depth.png    Depth 컬러맵
    <name>_cloud.ply    3D 포인트클라우드 (MeshLab/CloudCompare/Open3D 에서 열람)
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

import depth_to_3d as d3
import visualize_3d as viz

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def process(image: np.ndarray, name: str, step: int = 3, z_scale: float = 80.0,
            invert: bool = False) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    gray = d3.to_grayscale(image)
    depth = d3.estimate_depth(gray, invert=invert, blur_ksize=5)

    pts = d3.depth_to_pointcloud(depth, step=step, z_scale=z_scale)
    cols = d3.pointcloud_colors(image, step=step)

    d3.imwrite_unicode(os.path.join(OUT_DIR, f"{name}_depth.png"), d3.colorize_depth(depth))
    d3.save_ply(os.path.join(OUT_DIR, f"{name}_cloud.ply"), pts, cols)
    viz.render_panels(image, depth, os.path.join(OUT_DIR, f"{name}_panels.png"),
                      step=step, z_scale=z_scale,
                      title=f"2D -> 3D : {name}  (points={len(pts):,})")

    print(f"[{name}] gray {gray.shape} -> depth[{depth.min():.2f},{depth.max():.2f}] "
          f"-> {len(pts):,} points  (step={step}, z_scale={z_scale})")
    print(f"        saved: {name}_panels.png, {name}_depth.png, {name}_cloud.ply")


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        path = argv[1]
        img = d3.imread_unicode(path)
        if img is None:
            print(f"이미지를 읽지 못했습니다: {path}")
            return 1
        # 너무 크면 리사이즈(포인트 수/렌더 시간 관리)
        if max(img.shape[:2]) > 512:
            scale = 512 / max(img.shape[:2])
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        name = os.path.splitext(os.path.basename(path))[0]
        process(img, name)
    else:
        # 합성 장면 데모: 정답 형상(반구+램프)을 알고 있어 복원 검증에 좋다
        scene = d3.generate_synthetic_scene(200)
        scene_bgr = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
        process(scene_bgr, "synthetic", step=3, z_scale=80.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
