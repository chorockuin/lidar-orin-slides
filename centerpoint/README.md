# CenterPoint on NVIDIA AGX Orin — 구조·원리 웹 슬라이드

NVIDIA [Lidar_AI_Solution / CUDA-CenterPoint](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-CenterPoint) 모델이
LiDAR 포인트 클라우드를 입력받아 3D 바운딩 박스를 출력하기까지의 과정을 단계별로 설명하는 인터랙티브 HTML 슬라이드입니다.

## 보는 방법

`index.html`을 브라우저로 열고, 클릭 또는 `→` / `Space`로 한 단계씩 진행합니다. `←`로 되돌아갑니다.

## 파일

- `index.html` — 웹 슬라이드
- `requirements.md` — 요구사항 정의서

## 수정 방법

슬라이드는 `src/`의 세 파일로 나뉘어 있고, 빌드하면 `index.html`이 만들어집니다.

- `src/head.html` — 스타일
- `src/body.html` — 슬라이드 내용 (22장, 각 장의 자습 노트 포함)
- `src/script.html` — 슬라이드 엔진과 애니메이션

```sh
centerpoint/src/build.sh   # 저장소 루트에서, 또는 이 폴더에서 src/build.sh
```
