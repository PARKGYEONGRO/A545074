from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import NUM_CLASSES, DEVICE
from src.losses import Get_Loss


# 1. Loss Function Test
def Test_Loss_Function():
    Device = torch.device(DEVICE if DEVICE == 'cuda' or torch.cuda.is_available() else 'cpu')

    Prediction = torch.randn(
        2,
        NUM_CLASSES,
        384,
        384,
        device=Device,
        requires_grad=True
    )

    Target = torch.randint(
        0,
        NUM_CLASSES,
        (2, 384, 384),
        device=Device
    )

    for Loss_Type in ['ce', 'ce_dice']:
        Criterion = Get_Loss(Loss_Type=Loss_Type, Dice_Weight=1.0).to(Device)

        Loss = Criterion(Prediction, Target)

        assert torch.isfinite(Loss), (f'Non-finite loss: {Loss_Type}')

        print(f'{Loss_Type}: {Loss.item():.6f}')

    Combined_Criterion = Get_Loss(Loss_Type='ce_dice', Dice_Weight=1.0).to(Device)

    Combined_Loss = Combined_Criterion(Prediction, Target)

    Combined_Loss.backward()

    assert Prediction.grad is not None
    assert torch.isfinite(Prediction.grad).all()

    return True


# 2. Main
if __name__ == '__main__':
    Result = Test_Loss_Function()

    print('====================\nFINAL RESULT')
    if Result:
        print('[PASS]\n====================')
    else:
        print('[FAIL]\n====================')
