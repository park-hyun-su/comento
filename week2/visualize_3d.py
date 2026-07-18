"""2D->3D 변환 결과 시각화 (matplotlib, GUI 없이 PNG 저장).

- render_panels: 원본 / Depth 컬러맵 / 3D 표면 / 3D 포인트클라우드를 한 장에.
Open3D 가 설치돼 있으면 인터랙티브 뷰어도 열 수 있으나(선택), 여기서는
설치 부담 없는 matplotlib 만으로 정적 결과 이미지를 만든다.
"""
from __future__ import annotations

import cv2
import matplotlib
matplotlib.use("Agg")  # 창 없이 파일로만 저장
import matplotlib.pyplot as plt
import numpy as np

import depth_to_3d as d3


def render_panels(image: np.ndarray, depth: np.ndarray, out_path: str,
                  step: int = 3, z_scale: float = 80.0, title: str = "") -> None:
    """4분할 결과 이미지를 저장한다."""
    gray = d3.to_grayscale(image)
    depth_color = d3.colorize_depth(depth)                 # BGR
    pts = d3.depth_to_pointcloud(depth, step=step, z_scale=z_scale)
    cols = d3.pointcloud_colors(image, step=step) / 255.0

    h, w = depth.shape
    rows_i = np.arange(0, h, step)
    cols_i = np.arange(0, w, step)
    XX, YY = np.meshgrid(cols_i, rows_i)
    ZZ = depth[np.ix_(rows_i, cols_i)] * z_scale

    fig = plt.figure(figsize=(12, 9))
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else gray,
               cmap=None if image.ndim == 3 else "gray")
    ax1.set_title("1) Input 2D image"); ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB))
    ax2.set_title("2) Depth map (JET)"); ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3.plot_surface(XX, YY, ZZ, cmap="viridis", linewidth=0, antialiased=True)
    ax3.set_title("3) 3D surface (Z=depth)")
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
    ax3.view_init(elev=55, azim=-60); ax3.invert_yaxis()

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    ax4.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=2, depthshade=True)
    ax4.set_title(f"4) Point cloud ({len(pts):,} pts)")
    ax4.set_xlabel("X"); ax4.set_ylabel("Y"); ax4.set_zlabel("Z")
    ax4.view_init(elev=55, azim=-60); ax4.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def try_open3d_view(ply_path: str) -> bool:
    """Open3D 가 있으면 인터랙티브 뷰어를 연다. 없으면 False."""
    try:
        import open3d as o3d
    except ImportError:
        return False
    pcd = o3d.io.read_point_cloud(ply_path)
    o3d.visualization.draw_geometries([pcd])
    return True
