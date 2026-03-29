# GAR: Graph Assisted Reasoning for Object Detection

> **논문**: [GAR: Graph Assisted Reasoning for Object Detection (WACV 2020)](https://openaccess.thecvf.com/content_WACV_2020/papers/Li_GAR_Graph_Assisted_Reasoning_for_Object_Detection_WACV_2020_paper.pdf)
> Zheng Li, Xiaocong Du, Yu Cao (Arizona State University)

논문을 PyTorch로 재구현한 프로젝트입니다. Faster R-CNN (VGG-16) 위에 GCN 기반 그래프 추론 모듈을 추가하여 객체 간 관계 및 장면 정보를 활용한 객체 검출 성능 향상을 목표로 합니다.

**목표 성능**: PASCAL VOC 2007 test set mAP **76.1%** (baseline Faster R-CNN: 73.2%)

---

## 환경

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA RTX 5000 Ada Generation 32GB (**GPU 0번만 사용**) |
| CUDA | 12.2 (Driver 535.171.04) |
| Docker | PyTorch 2.1.0 + CUDA 12.1 (하위 호환) |
| Dataset | PASCAL VOC 2007 (trainval + test) |

---

## 프로젝트 구조

```
GAR-ObjectDetection/
├── Dockerfile                # PyTorch 2.1 + CUDA 12.1 기반 이미지
├── docker-compose.yml        # GPU 0, 볼륨 마운트, shm 설정
├── requirements.txt          # Python 의존성
├── configs/
│   └── gar_voc.yaml          # 모든 하이퍼파라미터
├── models/
│   ├── gar.py                # GARDetector - 전체 모델 (Faster R-CNN + GCR)
│   ├── gcr_module.py         # GCRModule - 2-layer GCN 그래프 추론
│   └── scene_detector.py     # SceneDetector - Places365 VGG-16 장면 검출
├── utils/
│   ├── voc_dataset.py        # VOC 데이터 로더 + 커스텀 transform
│   └── cooccurrence.py       # 공출현 행렬 오프라인 계산
├── train.py                  # Two-stage 학습 스크립트
├── evaluate.py               # VOC 2007 mAP 평가 스크립트
├── visualize/
│   ├── vis_cooccurrence.py   # 공출현 행렬 히트맵
│   ├── vis_graph.py          # GCR 그래프 구조 시각화
│   └── vis_detection.py      # 탐지 결과 비교 시각화
├── outputs/                  # (git 제외) 시각화 결과 저장
├── data/                     # (git 제외) 데이터셋 저장 위치
│   ├── VOCdevkit/VOC2007/
│   └── cooccurrence/         # 계산된 공출현 행렬 (.npy)
└── checkpoints/              # (git 제외) 학습된 모델 가중치
```

---

## 시작하기

### 1. 서버에서 레포 클론

```bash
git clone https://github.com/bk11052/GAR-ObjectDetection.git
```

### 2. VOC 2007 데이터셋 다운로드

```bash
mkdir -p data && cd data

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar

# tar 파일 정리 (선택)
rm -f *.tar

cd ..
```

다운로드 완료 후 `data/VOCdevkit/VOC2007/` 디렉토리가 생성됩니다.

### 3. Docker 이미지 빌드

```bash
docker compose build
```

최초 빌드 시 PyTorch 이미지 다운로드 때문에 시간이 걸릴 수 있습니다.

---

## 실행 방법

### Docker 컨테이너 진입

```bash
# GAR-ObjectDetection 폴더에서 실행
cd GAR/GAR-ObjectDetection

# 컨테이너 bash 셸 진입 (GPU 0 자동 할당)
docker compose run --rm gar bash
```

> 이후 모든 명령어는 **컨테이너 안에서** 실행합니다.
> 컨테이너 나가기: `exit` 입력

---

### Step 1. 공출현 행렬 계산 (최초 1회)

VOC 2007 trainval 데이터에서 객체-객체, 객체-장면 공출현 관계를 계산합니다.
Places365 사전학습 가중치를 자동 다운로드합니다 (~500MB).

```bash
python utils/cooccurrence.py \
    --voc_root data/VOCdevkit \
    --year 2007 \
    --split trainval \
    --save_dir data/cooccurrence/ \
    --device cuda:0
```

**소요 시간**: 약 20~30분
**결과물**: `data/cooccurrence/` 에 4개 `.npy` 파일 생성

| 파일 | 크기 | 설명 |
|------|------|------|
| `obj_obj_voc2007.npy` | 20x20 | 객체-객체 공출현 |
| `obj_inout_voc2007.npy` | 20x2 | 객체-실내/실외 |
| `obj_place_voc2007.npy` | 20x365 | 객체-장소 카테고리 |
| `obj_attr_voc2007.npy` | 20x102 | 객체-장면 속성 |

---

### Step 2. Stage 1 학습 — Faster R-CNN 백본 (4 epochs)

GCR 모듈 없이 기본 Faster R-CNN만 학습합니다.

```bash
python train.py --stage 1
```

| 항목 | 값 |
|------|------|
| Learning Rate | 5e-4 |
| Epochs | 5 |
| Batch Size | 1 |
| 학습 대상 | backbone + RPN + box_head + box_predictor |
| 동결 | scene_detector, gcr, scene_node_embed |

**결과물**: `checkpoints/stage1_best.pth`

---

### Step 3. Stage 2 학습 — GAR 전체 (6 epochs)

Stage 1 가중치를 로드한 뒤, GCR 그래프 추론 모듈을 포함하여 joint 학습합니다.

```bash
python train.py --stage 2 --resume checkpoints/stage1_best.pth
```

| 항목 | 값 |
|------|------|
| Learning Rate | 5e-5 |
| Epochs | 5 |
| 학습 대상 | rpn + box_head + box_predictor + gcr + scene_node_embed |
| 동결 | backbone (VGG-16 conv), scene_detector |

**결과물**: `checkpoints/stage2_best.pth`

---

### Step 4. 평가

VOC 2007 test set에서 per-class AP 및 mAP를 계산합니다.

```bash
# GAR (Stage 2) 평가
python evaluate.py --checkpoint checkpoints/stage2_best.pth --stage 2

# Baseline (Stage 1) 평가
python evaluate.py --checkpoint checkpoints/stage1_best.pth --stage 1
```

출력 예시:
```
============================================================
Class                      AP
------------------------------------------------------------
aeroplane               77.40%
bicycle                 81.30%
...
============================================================
mAP                     76.10%
============================================================
```

---

## 시각화

학습 완료 후 3가지 시각화를 생성합니다. 모두 **컨테이너 안에서** 실행합니다.

### 공출현 행렬 히트맵

객체-객체, 객체-실내외, 객체-장소, 객체-속성 관계를 히트맵으로 시각화합니다.

```bash
python visualize/vis_cooccurrence.py --cooc_dir data/cooccurrence/ --output_dir outputs/
```

**결과물**:
- `outputs/cooc_obj_obj.png` — 20x20 객체-객체 공출현 히트맵
- `outputs/cooc_obj_inout.png` — 20x2 객체-실내/실외 히트맵
- `outputs/cooc_obj_place_top10.png` — 가장 빈번한 장소 10개와의 관계
- `outputs/cooc_obj_attr_top10.png` — 가장 빈번한 속성 10개와의 관계

### GCR 그래프 구조

특정 이미지에 대한 GCR 모듈의 이종 그래프를 시각화합니다.
파란 원은 객체 노드, 초록 사각형은 장면 노드입니다.

```bash
python visualize/vis_graph.py \
    --checkpoint checkpoints/stage2_best.pth \
    --image data/VOCdevkit/VOC2007/JPEGImages/000001.jpg \
    --output_dir outputs/
```

**결과물**: `outputs/graph_000001.png` (원본 이미지 + 그래프 구조 나란히)

### 객체 탐지 결과 비교

Stage 1(Baseline)과 Stage 2(GAR)의 탐지 결과를 나란히 비교합니다.

```bash
# Baseline vs GAR 비교 (10장 랜덤)
python visualize/vis_detection.py \
    --checkpoint_s1 checkpoints/stage1_best.pth \
    --checkpoint_s2 checkpoints/stage2_best.pth \
    --output_dir outputs/ \
    --num_images 10

# GAR 결과만 보기
python visualize/vis_detection.py \
    --checkpoint_s2 checkpoints/stage2_best.pth \
    --output_dir outputs/ \
    --num_images 10
```

**결과물**: `outputs/detection_000001.png` ... (이미지별 bbox + 클래스 + confidence score)

---

## 핵심 아키텍처 요약

```
이미지 → VGG-16 backbone → feature map
            │
            ├── RPN → ROI Pooling → Box Head (4096-dim) → cRCN (cursory scores)
            │
            └── Scene Detector (Places365, frozen) → 장면 라벨 (469-dim)
                    │
                    └── Scene Node Embedding (S개 노드)
                            │
                            └── GCR Module (2-layer GCN)
                                    │
                                    ├── Instance edges: 객체-객체 공출현 기반
                                    ├── Scene edges: 객체-장면 공출현 기반
                                    └── Score Fusion: Z = wb·Yb + wg·Yg
                                            │
                                            └── 최종 cogitative scores → NMS → 검출 결과
```

**핵심**: 기존 Faster R-CNN의 cursory score에 GCN으로 추론한 graph score를 학습 가능한 가중치(wb, wg)로 융합하여 문맥에 맞지 않는 오탐을 억제하고, 관련 객체의 검출을 보강합니다.

---

## Quick Reference (명령어 요약)

```bash
# Docker 컨테이너 진입
cd GAR/GAR-ObjectDetection
docker compose run --rm gar bash

# 컨테이너 안에서 전체 파이프라인
python utils/cooccurrence.py                                          # Step 1 (최초 1회)
python train.py --stage 1                                             # Step 2
python train.py --stage 2 --resume checkpoints/stage1_best.pth       # Step 3
python evaluate.py --checkpoint checkpoints/stage2_best.pth --stage 2 # Step 4

# 시각화
python visualize/vis_cooccurrence.py                                  # 공출현 행렬 히트맵
python visualize/vis_graph.py --checkpoint checkpoints/stage2_best.pth --image data/VOCdevkit/VOC2007/JPEGImages/000001.jpg  # 그래프
python visualize/vis_detection.py --checkpoint_s1 checkpoints/stage1_best.pth --checkpoint_s2 checkpoints/stage2_best.pth    # 탐지 비교

# 컨테이너 나가기
exit
```

---

## 참고

- **GPU**: docker-compose.yml에서 `NVIDIA_VISIBLE_DEVICES=0`으로 GPU 0번만 사용
- **데이터**: `data/` 폴더는 `.gitignore`로 git에서 제외 (서버에서 직접 다운로드)
- **체크포인트**: `checkpoints/` 폴더도 git에서 제외
- **Places365 가중치**: 최초 실행 시 `~/.cache/places365/`에 자동 다운로드됨 (docker volume으로 유지)
