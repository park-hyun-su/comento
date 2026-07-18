# 2차 업무 — Unit Test 구성 및 2D → 3D 변환

Python `pytest`로 픽셀 처리·3D 변환 코드를 검증하고, OpenCV + NumPy로 2D 이미지를
**Depth Map → 3D 포인트클라우드**로 변환한다. 결과는 정적 이미지와 `.ply`로 저장한다.

## 실행

```bash
pip install -r requirements.txt

python -m pytest -v                 # 유닛테스트 16개
python run_pipeline.py              # 합성 장면(반구+램프) 데모
python run_pipeline.py <이미지경로>  # 실제 이미지로 변환
```

## 파일

| 파일 | 역할 |
|---|---|
| `depth_to_3d.py` | 핵심 로직 — 그레이스케일/정규화/Depth 추정/포인트클라우드/PLY 저장 (전부 순수 함수) |
| `visualize_3d.py` | matplotlib 4분할 시각화(원본·Depth·3D표면·포인트클라우드) |
| `run_pipeline.py` | 엔드투엔드 실행 진입점 |
| `tests/test_depth_to_3d.py` | pytest 유닛테스트 16개 |
| `outputs/` | 결과 이미지 및 `.ply` |

## 1. 왜 멘토 예제를 그대로 쓰지 않았나

업무요청서의 예제는 `cv2.applyColorMap(gray, COLORMAP_JET)`을 "Depth Map"이라 부른다.
하지만 이건 **밝기에 색만 입힐 뿐** 3차원 정보를 만들지 못한다 — 컬러맵을 씌우기 전과
후의 형상 정보량은 같다. 그래서 여기서는 **픽셀 밝기를 높이(Z)로 해석하는
height-field 방식**으로 실제 3D 좌표 `(X, Y, Z)`를 만든다.

```
X = 열(col),  Y = 행(row),  Z = normalize(밝기) × z_scale
```

이건 컴퓨터비전에서 **Shape-from-Shading**(밝기 → 표면 높이, Horn 1970)의 가장 단순한
형태다. 색은 시각화(`COLORMAP_JET`)에만 쓰고, 3D 복원은 밝기값 자체로 한다.

## 2. Unit Test — 무엇을 검증했나

`cv2.applyColorMap` 출력의 shape/type만 확인하는 예제 테스트는 "함수가 죽지 않는다"
이상을 보장하지 못한다. 대신 **수치 정답이 있는** 케이스로 로직을 검증했다.

| 테스트 그룹 | 검증 내용 |
|---|---|
| `to_grayscale` | shape·dtype, BT.601 known value(141), 이미 그레이면 통과, None→ValueError |
| `normalize01` | 범위 [0,1], **균일 이미지 0 나눗셈 방지** 분기 |
| `estimate_depth` | 범위 [0,1], `invert`가 밝기 극성을 뒤집고 `normal+invert==1` |
| `depth_to_pointcloud` | 점 개수 = H×W, step 서브샘플 개수, **Z = depth×z_scale 좌표 대응**, dtype |
| `pointcloud_colors` | 점–색 1:1 정렬, BGR→RGB 변환 |
| `save_ply` | PLY 헤더·정점 개수 정확성 |
| `backproject_pinhole` | 주점에서 X=Y=0, Z=depth (핀홀 역투영 공식) |

```
16 passed in 0.88s
```

전수 통과. 특히 `test_pointcloud_z_equals_depth_times_scale`은 (열,행)별 Z값이
`depth×z_scale`와 정확히 일치하는지 좌표 단위로 확인해, 변환 공식 자체를 못 박는다.

## 3. 2D → 3D 결과

### 합성 장면 (정답 형상을 아는 검증)
`outputs/synthetic_panels.png` — 반구(hemisphere) + 대각 램프(ramp). 정답 높이를 알고
있어 복원이 맞는지 눈으로 확인할 수 있다. 3D 표면에서 반구가 배경 램프 위로 정확히
솟아오른다. **알고리즘이 의도대로 동작함을 보증하는 기준점.**

### 실제 이미지 (티라미수)
`outputs/tiramisu_photo_panels.png` — 밝은 생크림은 높게, 어두운 초콜릿·유리컵은 낮게
복원된다. 형상이 그럴듯하게 잡히지만, 아래 한계가 그대로 드러난다.

## 4. 한계와 개선점 (결과 분석)

이 방식은 **밝기를 깊이로 가정**한다. 이 가정이 깨지는 지점이 곧 개선 과제다.

| 한계 | 증상 (티라미수) | 개선 방향 |
|---|---|---|
| 밝기 ≠ 실제 깊이 | 위에 얹힌 **어두운 초콜릿이 움푹 파인 구멍**으로 복원됨 | 실제 깊이 단서 필요 → 스테레오 / 학습기반 depth |
| 조명 의존 | 그림자·하이라이트가 가짜 높낮이를 만듦 | 알베도-조명 분리, 또는 조명 불변 depth 모델 |
| 절대 스케일 없음 | Z는 상대값(`z_scale`는 임의) | 스테레오 캘리브레이션 / metric depth(ZoeDepth) |
| 노이즈에 민감 | 텍스처가 표면을 우툴두툴하게 만듦 | 가우시안 평활화(적용함) + bilateral / 양방향 필터 |

**다음 단계로 갈 두 경로**
1. **스테레오** — `cv2.StereoSGBM`(Hirschmüller SGM) → disparity → `cv2.reprojectImageTo3D(Q)`.
   `backproject_pinhole()`이 이 경로의 역투영 부분을 미리 구현해 둔 것.
2. **단일 이미지 딥러닝 depth** — MiDaS / Depth Anything V2를 붙이면 캘리브레이션 없이
   진짜 상대깊이를 얻고, 같은 `depth_to_pointcloud()`로 포인트클라우드를 만들 수 있다.

## 5. 참고 논문 (2D → 3D 변환)

업무요청서 "참고 내용 — 2D → 3D 변환 알고리즘 및 관련 논문 참고" 항목.

**단일 이미지 → 깊이 (이 과제의 height-field와 가장 가까운 계열)**
- Eigen, Puhrsch, Fergus, *Depth Map Prediction from a Single Image using a Multi-Scale Deep Network*, NeurIPS 2014 — 단일 이미지 depth 회귀의 출발점. arXiv:1406.2283
- Ranftl et al., *MiDaS: Towards Robust Monocular Depth Estimation*, TPAMI 2022 — 여러 데이터셋 혼합 + scale-shift-invariant loss로 zero-shot 상대깊이. arXiv:1907.01341
- Yang et al., *Depth Anything V2*, NeurIPS 2024 — 대규모 pseudo-label로 학습한 depth 파운데이션 모델. arXiv:2406.09414
- **Horn, *Shape from Shading*, MIT 1970** — 밝기 → 표면 높이. 본 과제 height-field의 이론적 뿌리.

**스테레오 / disparity (OpenCV StereoSGBM·reprojectImageTo3D의 근거)**
- Scharstein & Szeliski, *A Taxonomy and Evaluation of Dense Two-Frame Stereo*, IJCV 2002 — OpenCV 스테레오 함수의 사고 틀. DOI:10.1023/A:1014573219977
- Hirschmüller, *Semi-Global Matching (SGM)*, TPAMI 2008 — `cv2.StereoSGBM` 내부 알고리즘. DOI:10.1109/TPAMI.2007.1166

**다중뷰 / 신경망 3D**
- Schönberger & Frahm, *Structure-from-Motion Revisited (COLMAP)*, CVPR 2016
- Mildenhall et al., *NeRF*, ECCV 2020 / Kerbl et al., *3D Gaussian Splatting*, SIGGRAPH 2023
- Liu et al., *Zero-1-to-3*, ICCV 2023 / *TripoSR*, 2024 — 단일 이미지 → 3D 생성

**위성/원격탐사 2D → 3D (DSM/DEM) — 온보드 위성 AI 관심 분야와 직결**
- Mou & Zhu, *IM2HEIGHT: Height Estimation from Single Monocular Imagery*, 2018, arXiv:1802.10249 — 단일 위성영상 → 높이맵(nDSM)의 대표작
- Liu et al., *IM2ELEVATION*, Remote Sensing 2020 — 단일 항공/위성영상 → DSM 회귀
- de Franchis et al., *S2P: Satellite Stereo Pipeline*, ISPRS 2014 — 위성 스테레오 → DSM (SGM 기반)
- *2023 IEEE GRSS DFC Track 2* — 단일뷰 광학+SAR → 건물 높이 벤치마크
