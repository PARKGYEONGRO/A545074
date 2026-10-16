# KoMaP A545074 개발 환경 안내서

## 문서 목적

이 문서는 KoMaP A545074 프로젝트의 개발 환경과 테스트 이미지 추론 방법을 안내합니다.
현재 개발 PC에서 확인한 환경 정보를 기록한 문서입니다.

## 확인된 개발 PC 환경

| 항목 | 확인된 버전 또는 정보 |
|---|---|
| 운영체제 | Windows |
| Python | 3.14.6 |
| PyTorch | 2.14.0+cu132 |
| torchvision | 0.29.1+cu132 |
| segmentation-models-pytorch | 0.5.0 |
| PyTorch CUDA 빌드 | 13.2 |
| GPU | NVIDIA GeForce RTX 5070 |
| 패키지 의존성 검사 | `pip check` 통과 |

## 1. Python 설치 확인

PowerShell을 열고 `submission` 폴더로 이동한다. 아래 경로는 예시이므로 실제 폴더 위치에 맞게 수정한다.

```powershell
cd C:\Users\사용자이름\OneDrive\Desktop\submission
```

Python 설치 여부를 확인한다.

```powershell
python --version
```

## 2. 가상환경 생성

`submission` 폴더 내부에서 실행한다.

```powershell
python -m venv venv
```

정상적으로 완료되면 `submission/venv/` 폴더가 생성된다.

```text
submission/
├─ venv/
├─ src/
├─ scripts/
├─ data/
├─ outputs/
├─ requirements.txt
└─ ENVIRONMENT.md
```

## 3. 가상환경 활성화

`submission` 폴더로 이동한 후, 사용 중인 터미널에 맞는 명령어를 실행한다.

| 터미널 | 폴더 이동 명령어 | 가상환경 활성화 명령어 |
|---|---|---|
| PowerShell | `cd C:\Users\사용자이름\OneDrive\Desktop\submission` | `.\venv\Scripts\Activate.ps1` |
| 명령 프롬프트 (CMD) | `cd /d C:\Users\사용자이름\OneDrive\Desktop\submission` | `venv\Scripts\activate.bat` |
| Git Bash | `cd /c/Users/사용자이름/OneDrive/Desktop/submission` | `source venv/Scripts/activate` |

> **참고:** `사용자이름`과 폴더 경로는 실제 PC의 경로에 맞게 수정한다.

가상환경이 정상적으로 활성화되면 터미널 앞부분에 `(venv)`가 표시된다.

### 3-1. PowerShell 실행 정책으로 활성화가 차단되는 경우

현재 실행 정책을 확인한다.

```powershell
Get-ExecutionPolicy
Get-ExecutionPolicy -List
```

본인 계정에만 적용되는 실행 정책을 변경하려면 다음 명령어를 사용할 수 있다.

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

확인 메시지가 나타나면 내용을 확인한 후 승인하고, 다시 활성화한다.

```
PowerShell	.\venv\Scripts\Activate.ps1
명령 프롬프트 (CMD)	venv\Scripts\activate.bat
Git Bash	source venv/Scripts/activate
Git Bash (동일한 방식)	venv/Scripts/activate
```

정책 변경이 허용되지 않는 경우에는 CMD 또는 Git Bash에서 활성화를 시도한다.

경로는 실제 폴더 위치에 맞게 수정한다.

## 4. 필요한 라이브러리 설치 및 모델 가중치 배치

### 4-1. 필요한 라이브러리 설치

가상환경이 활성화된 상태인지 확인한다.

```powershell
python --version
python -m pip --version
```

pip를 업데이트한다.

```powershell
python -m pip install --upgrade pip
```

프로젝트 패키지를 설치한다.

```powershell
python -m pip install -r requirements.txt
```

설치 후 의존성 오류를 확인한다.

```powershell
python -m pip check
```

오류가 발생하면 메시지를 확인하고 해당 패키지의 버전 및 설치 조건을 점검한다.

### 4-2. 모델 가중치 파일 배치

모델 가중치 파일(`exp011_final_model.pth`)은 제출 ZIP에 포함되지 않으며, 별도 이메일로 제출된다.

가중치 파일을 전달받은 후 프로젝트 루트 기준으로 다음 경로에 배치한다.

```text
submission/
└─ outputs/
   └─ final/
      └─ exp011_final_model.pth
```

파일명과 경로가 정확한지 확인한다. 가중치 파일이 올바르게 배치되어야 다음 단계의 테스트 이미지 추론을 실행할 수 있다.

## 5. 테스트 이미지 추론 실행

`submission` 폴더를 작업 디렉터리로 유지하고, 가상환경이 활성화된 상태에서 실행한다.

```powershell
python -m scripts.predict_test
```

클래스 ID 마스크는 다음 위치에 저장된다.

```text
submission/
└─ outputs/
   └─ final/
      └─ test_predictions/
```

## 6. RGB 컬러 마스크 변환

클래스 ID 마스크가 생성되면 실행한다.

```powershell
python -m scripts.convert_predictions_to_rgb
```

최종 RGB 컬러 마스크는 다음 위치에 저장된다.

```text
submission/
└─ outputs/
   └─ final/
      └─ test_predictions_rgb/
```

`test_predictions_rgb/` 폴더에 이미지 10장이 생성되었는지 확인한다.

## 7. 최종 확인 체크리스트

- [ ] Python 설치 및 버전 확인
- [ ] `venv` 가상환경 생성 및 활성화
- [ ] `requirements.txt` 라이브러리 설치
- [ ] `python -m pip check` 결과 확인
- [ ] 테스트 이미지 10장 추론 완료
- [ ] `outputs/final/test_predictions/`에 클래스 ID 마스크 생성
- [ ] `outputs/final/test_predictions_rgb/`에 RGB 컬러 마스크 10장 생성
"""

## 8. 실행 전 확인 사항

- 프로젝트 루트 폴더에서 명령어를 실행합니다.
- 최종 모델 가중치가 `submission/outputs/final/exp011_final_model.pth`에 있어야 합니다.
- 테스트 이미지가 `submission/data/test/images/`에 있어야 합니다.
- 필요한 Python 패키지가 설치되어 있어야 합니다.
- 결과물 이미지 제출 시에는 대회에서 요구하는 마스크 형식과 파일명 규칙을 확인합니다.

## 9. 환경 재현성에 관한 주의 사항

이 문서의 버전 정보는 현재 개발 PC에서 확인한 값입니다. 다른 PC에서 동일하게 실행되는지는 별도 환경에서 설치 및 추론을 실행해 검증해야 합니다.

### 10. CPU 기반 추론 환경 구성

개발 PC에서는 NVIDIA RTX 5070 GPU와 CUDA를 활용하여 모델 학습 및 개발을 진행하였다. 그러나 제출 프로그램이 다양한 하드웨어 환경에서도 동일하게 실행될 수 있도록, 추론 단계에서는 GPU 대신 CPU를 사용하도록 구성하였다.

CPU 추론은 GPU 추론보다 실행 시간이 길어질 수 있으나, **CPU 추론은 GPU 추론보다 실행 시간이 길어질 수 있다. 동일한 모델 가중치와 추론 로직을 사용하며, 추론 및 평가 방식은 동일하게 유지한다. 다만 연산 환경에 따른 미세한 수치 차이로 일부 픽셀의 예측 결과가 달라질 가능성이 있다.** 이를 통해 특정 GPU 환경에 대한 의존성을 줄이고, 별도의 GPU가 없는 PC에서도 추론을 수행할 수 있도록 하였다.

## 11. 최종 모델 정보

- 모델 실험: EXP-011
- 검증 데이터 기준 최고 mIoU: `0.748260`
- 최종 모델 파일: `submission/outputs/final/exp011_final_model.pth`

위 mIoU는 검증 데이터에서 측정한 값이며, 정답 마스크가 없는 테스트 데이터의 실제 성능을 의미하지 않습니다.
