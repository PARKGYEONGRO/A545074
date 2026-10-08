import sys
from pathlib import Path

import torch

# 1. Project Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import NUM_CLASSES
from src.metrics import (
    Calculate_IoU,
    Calculate_mIoU,
    Calculate_Metrics
)


# 2. Device
DEVICE = torch.device('cuda'
    if torch.cuda.is_available()
    else 'cpu'
)


# 3. Metric Test
def Test_Metrics():
    print('====================\nKoMaP Metrics Test\n====================')
    print()

    print(f'[Device]: {DEVICE}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('GPU: Not Available')
    print()

    # Dummy Prediction
    Prediction = torch.randn(
        2,
        NUM_CLASSES,
        384,
        384,
        device=DEVICE
    )

    # Dummy Target
    Target = torch.randint(
        0,
        NUM_CLASSES,
        (2, 384, 384),
        device=DEVICE
    )

    print('[Input]')
    print(f'Prediction Shape: {tuple(Prediction.shape)}\n'
            f'Prediction Dtype: {Prediction.dtype}\n'
            f'Target Shape: {tuple(Target.shape)}\n'
            f'Target Dtype: {Target.dtype}')
    print()

    # IoU Test
    IoU_List = Calculate_IoU(Prediction, Target, NUM_CLASSES)
    print('[IoU Test]')
    for Class_ID, IoU in enumerate(IoU_List):
        if torch.isnan(IoU):
            print(f'Class {Class_ID} IoU: NaN')
        else:
            print(f'Class {Class_ID} IoU: {IoU.item():.6f}')
    print()

    # mIoU Test
    mIoU = Calculate_mIoU(Prediction, Target, NUM_CLASSES)
    print('[mIoU Test]')
    print(f'mIoU: {mIoU.item():.6f}')
    print()

    # Metric Dictionary Test
    Metrics = Calculate_Metrics(Prediction, Target, NUM_CLASSES)
    print('[Metric Dictionary Test]')
    print(f'IoU Type  : {type(Metrics['IoU'])}')
    print(f'mIoU Type : {type(Metrics['mIoU'])}')

    # Shape Test
    if IoU_List.shape == (NUM_CLASSES,):
        print(f'[PASS] IoU Shape: {tuple(IoU_List.shape)}')
    else:
        print(f'[FAIL] IoU Shape: {tuple(IoU_List.shape)}')
        return False

    # mIoU Test
    if mIoU.ndim == 0:
        print('[PASS] mIoU Shape: Scalar')
    else:
        print('[FAIL] mIoU Shape: Expected Scalar')
        return False

    # Value Test
    if (torch.isfinite(mIoU) and 0.0 <= mIoU.item() <= 1.0):
        print('[PASS] mIoU Value: Valid')
    else:
        print('[FAIL] mIoU Value: Invalid')
        return False


    print()
    return True


# 4. Main
if __name__ == '__main__':
    Result = Test_Metrics()

    print('====================\nFINAL RESULT')
    if Result:
        print('[PASS]\n====================')
    else:
        print('[FAIL]\n====================')
