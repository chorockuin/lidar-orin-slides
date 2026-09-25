# CenterPoint (LiDAR 3D 객체 검출) 구조·원리 설명 웹 슬라이드 — 요구사항 정의서

> 상태: **확정 v1** · 작성일: 2026-09-25
> 확인 질문(8장)에 대한 사용자 답변을 반영함.

---

## 1. 목적

NVIDIA Jetson **AGX Orin**에서 동작하는 **CenterPoint LiDAR 공식 모델**의 구조와 동작 원리를,
**입력(포인트 클라우드) → 출력(3D 바운딩 박스)** 까지 하나의 흐름으로 이해할 수 있는 **HTML 웹 슬라이드**로 만든다.
학습 원리와 Orin 배포(TensorRT) 실무까지 포함한다.

## 2. 대상 독자 · 용도

- **용도: 자습 후 발표.** 슬라이드 자체만 읽어도 이해되는 자습 자료이면서, 그대로 발표에 쓸 수 있어야 한다.
  - 화면(발표용)에는 그림·핵심 문장만, 상세 설명은 펼쳐보기/노트 패널(자습용)로 분리한다.
- 딥러닝/LiDAR 인식 **초보자**가 끝까지 따라올 수 있어야 한다.
- 동시에 실제 구현을 다루는 엔지니어가 봐도 **텐서 shape·연산 수준까지 정확**해야 한다.
- → "직관(왜?) 먼저, 수식·shape(어떻게?)는 그다음" 순서로 층을 나눠 설명한다.

## 3. 대상 모델 (확정)

- **NVIDIA 공식 `Lidar_AI_Solution / CUDA-CenterPoint`** (TensorRT 기반, AGX Orin 대상 배포 예제)
- 데이터셋: **nuScenes** (10클래스, 6개 task group)
- 원 모델: CenterPoint (Yin et al., CVPR 2021) VoxelNet 0.075 voxel 설정

아래 수치는 제작 전 공식 저장소 소스/설정 파일 기준으로 재검증하여 반영한다.

| 단계 | 구성 | 주요 사양 (재검증 예정) |
|---|---|---|
| 입력 | nuScenes LiDAR 포인트 (여러 sweep 누적) | 점당 5채널: x, y, z, intensity, Δt |
| 전처리 | GPU 복셀화 (CUDA 커널) | 범위 [-54, 54]×[-54, 54]×[-5, 3] m, 복셀 0.075×0.075×0.2 m → 격자 1440×1440×40(41), 복셀 내 점 평균 |
| 백본 | 3D Sparse Convolution (`SpMiddleResNetFHD`) | libspconv, SubM conv + stride‑2 sparse conv로 ×8 다운샘플 |
| 높이 압축 | Sparse → Dense, Z축을 채널로 펼침 | (C, D, H, W) → (C·D, H, W) ≈ 256×180×180 BEV |
| 넥 | RPN (SECOND-FPN 스타일) 2D CNN | 다중 스케일 conv + deconv 후 concat |
| 헤드 | CenterHead (6 task group) | heatmap + reg(2), height(1), dim(3), rot(2), vel(2) |
| 후처리 | GPU 디코딩 + NMS | peak → 박스 복원, score threshold, NMS |
| 출력 | 3D 박스 목록 | (x, y, z, w, l, h, yaw, vx, vy, score, class) |
| 배포 | TensorRT FP16 / INT8 on AGX Orin | 단계별 latency |

## 4. 콘텐츠 요구사항

### 4.1 전체 흐름 (필수)
- 초반에 **전체 파이프라인 지도**를 보여주고, 이후 모든 슬라이드 상단에 "현재 위치"를 표시한다.
- 각 단계마다 반드시:
  1. **무엇을 하는가** (동작)
  2. **무엇을 의미하는가** (입력 shape → 출력 shape)
  3. **왜 그렇게 설계했는가** (대안과 비교)

### 4.2 슬라이드 구성 (**20장 내외**, 각 장 안에서 클릭으로 여러 단계 진행)

| # | 제목 | 핵심 내용 |
|---|---|---|
| 1 | 표지 | 제목, 한 줄 요약 |
| 2 | 문제 정의 | LiDAR·포인트 클라우드란? 3D 객체 검출의 입력과 출력 |
| 3 | 핵심 아이디어 | Anchor 기반 vs **Center 기반** — 왜 물체를 "점"으로 찾는가 |
| 4 | 전체 파이프라인 지도 | 입력→복셀화→3D 백본→BEV→Neck→Head→후처리, 각 단계 shape |
| 5 | 입력 데이터 | 점 1개의 5채널, multi-sweep 누적과 Δt의 의미 `[N, 5]` |
| 6 | 복셀화 ① 격자 | `idx = floor((p − min)/voxel)` 토이 예제, 격자 크기 계산 |
| 7 | 복셀화 ② 특징 | 복셀 내 평균, 최대 개수 제한, 희소성(≈ 수 %만 채워짐) |
| 8 | Sparse Conv ① 원리 | Dense vs Sparse, SubM vs 일반 sparse conv 애니메이션 |
| 9 | Sparse Conv ② 행렬 연산 | rulebook: gather → `[N,Cin]×[Cin,Cout]` → scatter-add |
| 10 | 3D 백본 구조 | 스테이지별 shape 변화 블록 다이어그램 |
| 11 | 높이 압축 → BEV | Z축을 채널로 눕히기 `[C,D,H,W] → [C·D,H,W]` |
| 12 | Neck (RPN) | 다운/업샘플, concat, receptive field |
| 13 | CenterHead 구조 | 공유 conv + task group별 헤드, 출력 채널 표 |
| 14 | Heatmap의 의미 | 가우시안 peak, 클래스별 채널 |
| 15 | 회귀 헤드의 의미 | offset, z, log-dim, sin/cos yaw, velocity — 왜 이렇게 인코딩? |
| 16 | 학습 ① 정답 만들기 | 가우시안 반지름, heatmap 타깃 생성 |
| 17 | 학습 ② Loss | Focal loss(왜 필요한가), L1 회귀 loss, 가중치 |
| 18 | 후처리 | peak 찾기, 디코딩 수식(격자→미터), NMS |
| 19 | Orin 배포 ① 개념 | ONNX → TensorRT, FP16/INT8 양자화가 무엇이고 왜 빠른가 |
| 20 | Orin 배포 ② 실무 | CUDA-CenterPoint 구조(전처리 커널 / libspconv / TRT 엔진 / 후처리 커널), 엔진 빌드, 단계별 latency, 메모리 |
| 21 | 정리 | 한 장 요약 + 용어집 |

### 4.3 수학·shape 설명 (필수)
- 모든 주요 텐서의 **shape 표기** (예: `[N_points, 5]`, `[N_voxels, 5]`, `[1, 256, 180, 180]`).
- 핵심 연산은 **행렬 연산 형태로 풀어서** 보여주고, 작은 숫자 예시(toy example)에 실제 값을 대입해 한 번 더 보여준다.
  - 좌표 → 복셀 인덱스, sparse conv gather‑GEMM‑scatter, 박스 디코딩, 가우시안 타깃, focal loss.

## 5. 시각·인터랙션 요구사항

- **형식**: 단일 HTML 파일(브라우저에서 바로 열림, 설치 불필요).
- **단계별 진행**: 클릭 / →, Space 키마다 **한 단계씩** 요소 등장·애니메이션 진행. ← 키로 되돌아가기.
- **동적 시각화**
  - 포인트가 격자에 떨어져 복셀로 묶이는 애니메이션
  - 커널이 dense vs sparse 방식으로 움직이는 비교
  - 텐서 블록(직육면체)이 단계마다 크기·채널이 변하는 모습
  - 높이 방향을 채널로 "눕히는" 애니메이션
  - heatmap peak가 떠오르고 박스가 자라나는 애니메이션
  - 행렬 곱 셀 단위 하이라이트
- 진행 표시(현재 슬라이드/단계), 목차 바로가기, 자습용 상세 설명 패널(펼쳐보기).
- 다크/라이트 모드 가독성, 16:9 발표 화면 기준.

## 6. 품질 요구사항

- 수치는 **공식 저장소 설정값 기준**. 불확실한 값은 "예시값"으로 명시.
- 용어는 처음 등장 시 한 줄 정의.
- 언어: **한국어** (기술 용어는 영어 병기).

## 7. 결과물 · 버전 관리

- 로컬 `index.html` + **claude.ai 비공개 링크**로 게시.
- GitHub **공개(public)** 저장소 `centerpoint-orin-slides` 에 push. (GitHub Pages는 사용하지 않음)
- 폴더 구조:
  ```
  lidar/
  ├── README.md          # 프로젝트 소개, 보는 방법
  ├── requirements.md    # 이 문서
  └── index.html         # 웹 슬라이드 (단일 파일)
  ```
- 작업 단위(요구사항 확정, 슬라이드 완성 등)마다 커밋한다.

## 8. 확인 사항 답변 기록

| 질문 | 답변 |
|---|---|
| Q1 대상 모델 | NVIDIA Lidar_AI_Solution / CUDA-CenterPoint |
| Q2 데이터셋 | nuScenes |
| Q3 학습 파트 | 포함 |
| Q4 배포 깊이 | 개념은 쉽게 충분히 + 실무 수준까지 |
| Q5 결과물 | 로컬 HTML + claude.ai 비공개 링크 |
| Q6 분량/용도 | 20장 내외, 자습 후 발표 |
| Q7 GitHub | 이름은 추천안(`centerpoint-orin-slides`), 공개, Pages 미사용, gh로 생성·push |
