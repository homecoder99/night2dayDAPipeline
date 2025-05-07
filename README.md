# 🌓 Night2Day DA Pipeline

`Night2Day`는 야간 도로 객체 인식을 개선하기 위한 **CycleGAN 기반 이미지 도메인 전환 + YOLOv8 도메인 적응 학습 파이프라인**입니다.

> ✅ 기반 프레임워크: [Ultralytics YOLOv8](https://docs.ultralytics.com)
> ✅ 데이터셋: [BDD100K](https://bdd-data.berkeley.edu/) (야간 객체 탐지)
> ✅ 구현 언어: Python 3.11+
> ✅ 지원 플랫폼: macOS M1/M2 (MPS), CUDA, CPU

---

## 📁 프로젝트 구조

```
nightvision/
├── main.py                             # 환경 변수 정의
├── night2day.py                        # 전체 파이프라인 정의 (GAN + YOLOv8 DA)
├── dataset.py                          # BDD100K → YOLO 형식 변환기
├── yolov8n.yaml                        # YOLOv8 모델 구조 정의 (커스텀 클래스 수 포함)
├── bdd100k_yolo_night.yaml             # 데이터셋 구성 YAML
├── pyproject.toml                      # Python 환경 및 패키지 정의
└── README.md                           # 프로젝트 문서
```

---

## 🔧 주요 구성 요소 및 기능

### 1. `dataset.py`

- BDD100K의 2D detection json 라벨을 **YOLO 포맷(txt)** 으로 변환
- `convert_split()`: 전체 데이터(train/val/test)
- `convert_night()`: 야간 이미지만 필터링
- `create_yolo_yaml()`: 학습용 `.yaml` 자동 생성

### 2. `night2day.py`

- **CycleGAN 학습**

  - `train_cyclegan(data_dir)`: Night ↔ Day 변환용 CycleGAN 학습

- **이미지 증강 (Night→Day)**

  - `augment_with_generator(...)`: 학습된 GAN으로 이미지 변환

- **YOLOv8 Domain Adaptation 학습**

  - `YOLOv8_DA`: 도메인 분류기 포함 YOLOv8 커스텀 클래스
  - `train_yolov8_da(...)`: 학습 루프 + DetectionTrainer 활용

### 3. `yolov8n.yaml`, `bdd100k_yolo_night.yaml`

- `yolov8n.yaml`: YOLOv8n 구조 정의 + `nc: 10` 설정
- `bdd100k_yolo_night.yaml`: 학습용 이미지/라벨 경로, 클래스 정의 포함

---

## 🚀 실행 순서

### 1. 가상환경 세팅, 환경 변수 설정

```bash
# 프로젝트 루트 기준
pip install -r requirements.txt
# 또는 pyproject.toml 기반
uv pip install -r pyproject.toml
# main.py에 있는 bash 명령어 실행
export ...
```

### 2. BDD100K → YOLO 데이터셋 변환

```bash
python -m dataset
```

- `bdd100k_yolo_night.yaml` 자동 생성됨

### 3. CycleGAN 학습 (선택)

```bash
python night2day.py --mode train_gan --data_dir /path/to/cyclegan_data --epochs 100
```

### 4. Night → Day 이미지 증강 (선택)

```bash
python night2day.py --mode augment \
  --generator_ckpt runs/cyclegan/G_A_e100.ckpt \
  --input_dir bdd100k_yolo_night/images/train \
  --output_dir bdd100k_yolo_day/images/train
```

### 5. YOLOv8 도메인 적응 학습

```bash
python night2day.py \
  --mode train_yolo \
  --data_yaml your/path/bdd100k_yolo_night.yaml \
  --epochs 100 \
  --device mps  # or cuda
```

---

## 📁 데이터셋 디렉토리 구조

변환 후 최종 디렉토리 루트는 보통 다음과 같이 구성됩니다:

```
bdd100k_yolo_night/
├── images/
│   ├── train/
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   └── ...
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   │   ├── 0001.txt
│   │   ├── 0002.txt
│   │   └── ...
│   ├── val/
│   └── test/
└── bdd100k_yolo_night.yaml  ← 이 디렉토리 전체를 설명하는 YOLO 포맷 YAML
```

---

## 📄 라벨 파일 구조 (YOLO format)

- 각 이미지에 대응하는 `.txt` 파일 존재
- 각 줄은 하나의 객체 (object) 정보를 담고 있음

```
<class_id> <x_center> <y_center> <width> <height>
```

예시 (`labels/train/0001.txt`):

```
2 0.533 0.461 0.147 0.281  # class_id=2 (car)
0 0.478 0.402 0.095 0.221  # class_id=0 (person)
```

> 모든 값은 **0\~1 사이로 정규화된 비율**입니다.

---

## 📘 bdd100k_yolo_night.yaml 예시

```yaml
path: your/database/path
train: images/train
val: images/val
test: images/test

nc: 10
names:
  0: person
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motor
  7: bike
  8: traffic light
  9: traffic sign
```

- `path`: 루트 디렉토리
- `train`, `val`, `test`: 각각의 하위 폴더 경로
- `nc`: 클래스 수
- `names`: 클래스 ID ↔ 이름 매핑

---

## ✅ 요약: 최소 필요 조건

| 디렉토리                  | 내용                         |
| ------------------------- | ---------------------------- |
| `images/train/`           | 학습용 이미지 (`.jpg`)       |
| `labels/train/`           | 학습용 라벨 (`.txt`)         |
| `bdd100k_yolo_night.yaml` | 경로/클래스 정의용 메타 파일 |

## 📦 의존 패키지

```toml
# pyproject.toml 발췌:contentReference[oaicite:2]{index=2}
ultralytics >= 8.3.128
torch >= 2.7.0
torchvision >= 0.22.0
opencv-python >= 4.11.0
```

---

## 📌 참고

- Ultralytics 공식 문서: [https://docs.ultralytics.com](https://docs.ultralytics.com)
- BDD100K 데이터: [https://bdd-data.berkeley.edu](https://bdd-data.berkeley.edu)
- MPS(Mac) 사용 시 `--device mps` 옵션 필요

---

## 📮 기여 & 라이센스

- 코드 또는 이슈 제보는 PR로 환영합니다.
- 본 프로젝트는 MIT License 하에 배포됩니다.

---
