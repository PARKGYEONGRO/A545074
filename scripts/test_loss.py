import sys
from pathlib import Path

import torch

# 1. Project Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import NUM_CLASSES
from src.losses import Get_Loss


# 2. Device
DEVICE = torch.device('cuda'
    if torch.cuda.is_available()
    else 'cpu'
)


# 3. Loss Test
def Test_Loss():

    print('====================\nKoMaP Loss Test\n====================')
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

    # Loss Creation
    Loss_Function = Get_Loss()

    # Forward Test
    Loss = Loss_Function(Prediction, Target)


    print('[Loss Test]')
    print(f'Loss Value: {Loss.item():.6f}\n'
          f'Loss Dtype: {Loss.dtype}\n'
          f'Loss Shape: {tuple(Loss.shape)}')
    print()


    # Shape Test
    if Loss.ndim == 0:
        print('[PASS] Loss Shape: Scalar')
    else:
        print('[FAIL] Loss Shape: Expected Scalar')
        return False


    # Dtype Test
    if Loss.dtype == torch.float32:
        print('[PASS] Loss Dtype: torch.float32')
    else:
        print('[FAIL] Loss Dtype')
        return False


    # Value Test
    if torch.isfinite(Loss):
        print('[PASS] Loss Value: Finite')
    else:
        print('[FAIL] Loss Value: Invalid')
        return False


    print()
    return True

# 4. Main
if __name__ == '__main__':
    Result = Test_Loss()

    print('====================\nFINAL RESULT')
    if Result:
        print('[PASS]\n====================')
    else:
        print('[FAIL]\n====================')
