# LiDAR 3D 객체 검출 모델 — 구조·원리 웹 슬라이드

NVIDIA Jetson AGX Orin에서 동작하는 LiDAR 3D 객체 검출 모델을, 입력 포인트 클라우드에서 3D 바운딩 박스까지 단계별로 설명하는 인터랙티브 HTML 슬라이드 모음입니다. 모델마다 폴더가 따로 있습니다.

| 폴더 | 모델 | 상태 |
|---|---|---|
| [`centerpoint/`](centerpoint/) | NVIDIA [CUDA-CenterPoint](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-CenterPoint) (nuScenes) | 슬라이드 22장 완성 |
| [`pointpillars/`](pointpillars/) | NVIDIA [CUDA-PointPillars](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars) (KITTI) | 요구사항 초안 |

각 폴더에는 `requirements.md`(요구사항 정의서), `index.html`(웹 슬라이드), `src/`(슬라이드 원본과 빌드 스크립트)가 같은 구성으로 들어갑니다.
