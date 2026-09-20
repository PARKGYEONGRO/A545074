# 2026 KoMaP AI 경진대회

고탄성 주조용 알루미늄 합금의 OM 현미경 이미지에서 4개 소재 상을 픽셀 단위로 분할하는 시맨틱 세그멘테이션 프로젝트

## 목표

입력 현미경 이미지로부터 다음 4개 상의 마스크를 예측

- Primary Si
- Al3Ni
- Eutectic Si
- Al

모델 성능은 validation 데이터 20장에 대해 계산한 클래스별 IoU의 평균인 mIoU로 평가합니다. 최종 산출물은 test 이미지 10장의 예측 마스크

## 개발 환경

- Python 가상환경
- NVIDIA GeForce RTX 5070, VRAM 약 12GB
- PyTorch CUDA 환경
- VS Code를 주 개발 도구로 사용
- Jupyter Notebook은 EDA, 마스크·예측 시각화, 오류 분석에 사용
- Google Colab은 노트북에서 GPU가 필요한 큰 테스트를 위한 보조 환경으로만 사용

## 주요 라이브러리

- `torch`, `torchvision`: 모델 학습 및 추론
- `segmentation-models-pytorch`: U-Net, DeepLabV3+ 등 분할 모델
- `albumentations`: 이미지와 마스크 동시 증강
- `opencv-python`, `Pillow`: 이미지 입출력 및 처리
- `torchmetrics`: IoU, mIoU 등 평가 지표
- `PyYAML`: 학습·모델 설정 관리

초기에는 일반 PyTorch 학습 구조와 YAML 설정 파일을 사용합니다. Lightning, Hydra 등 대형 프레임워크는 기본 파이프라인이 안정화된 후 필요할 때 검토

## 프로젝트 구조

```text
.
├── configs/        # 모델·학습 설정 YAML
├── data/           # 대회 데이터 (Git 추적 제외)
├── notebooks/      # EDA·시각화·오류 분석용 .ipynb
├── src/            # 데이터셋, 모델, 손실함수, 평가 로직
├── scripts/        # 학습, 검증, 추론 실행 스크립트
├── checkpoints/    # 모델 가중치 (Git 추적 제외)
├── outputs/        # 예측 결과·시각화 (Git 추적 제외)
├── environment.yaml
└── README.md
```

## 개발 순서

1. 이미지·마스크 파일 대응, 해상도, RGB 색상값, 클래스 분포를 점검
2. 정답 마스크의 4개 색상을 정수 클래스 ID로 변환
3. train 데이터로 학습하고 validation 데이터로만 모델과 하이퍼파라미터를 선택
4. 기본 분할 모델을 학습하고 validation mIoU를 산출
5. test 이미지 10장에 대한 마스킹 결과를 생성
6. 기본 파이프라인이 안정화된 뒤 독창성 기능을 적용

## 모델링 원칙

- 기준선은 U-Net 계열, DeepLabV3+, SegFormer를 후보로 비교
- 손실함수는 Cross-Entropy를 기본으로 Dice, Focal 등의 보조 손실을 비교
- 마스크 리사이즈에는 nearest-neighbor 보간만 사용
- validation mIoU와 클래스별 IoU를 기준으로 모델 선택
- seed, 모델 설정, 입력 크기, batch size, learning rate, epoch, optimizer, scheduler를 실험마다 기록
- test 데이터는 최종 추론 전까지 모델 선택에 사용 X

## 독창성 기능

기본 모델 성능이 확보된 뒤 아래 기능을 실험

- 경계 인식 다중 과제 학습: 소재 상 경계의 오분류와 마스크 번짐을 줄임
- 미세조직 정량 분석: 상별 면적률, 입도, 입자 수, 형상 지표 등을 자동 산출
- 불확실성 지도: 전문가의 재검토가 필요한 영역을 시각화

## Git 관리

대회 데이터, 모델 가중치, 예측 결과, 제출 ZIP 파일은 저장소에 올리지 않습니다. 코드, 설정 파일, 실행 방법, 환경 YAML, 문서만 Private GitHub 저장소에서 관리