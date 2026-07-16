# comento

코멘토 직무부트캠프 — Computer Vision 과제 저장소

| 차수 | 주제 | 산출물 |
|---|---|---|
| [1차](week1/) | Git 활용 및 픽셀 단위 이미지 처리 | [README](week1/README.md) · [결과 이미지](week1/preprocessed_samples/) · [PPT](week1/1차업무_결과보고_박현수.pptx) |

## 브랜치 전략

기능 단위로 브랜치를 끊고 PR로 병합한다.

```
main
 ├── feature/image-processing   기본 전처리 + NumPy 대조 검증   PR #1
 ├── feature/color-detection    HSV 색상 감지 / 필터링          PR #2
 ├── feature/outlier-filter     이상치 탐지                     PR #3
 └── feature/pipeline-report    파이프라인 · 결과 · 문서        PR #4, #5
```
