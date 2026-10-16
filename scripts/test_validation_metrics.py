# Validation Metrics Test

import sys
from pathlib import Path

import torch

# 1. Project Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

# 2. Import
from src.config import NUM_CLASSES

from src.metrics import(
    Calculate_IoU,
    Calculate_mIoU,
    Calculate_Image_Metrics,
    Calculate_Validation_mIoU
)

# 3. Main Test
def Main():
    print('=' * 60)
    print('KoMaP KoMaP Validation Metrics Test')
    print('=' * 60)

    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\nDevice: {Device}')

    # Original Image Mask Test
    Target_Mask = torch.tensor(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 2],
            [2, 2, 3, 3],
            [2, 3, 3, 3]
        ],
        dtype=torch.long,
        device=Device
    )

    Prediction_Mask = Target_Mask.clone()

    Image_Result = Calculate_Image_Metrics(
        Prediction_Mask=Prediction_Mask,
        Target_Mask=Target_Mask,
        Num_Classes=NUM_CLASSES
    )

    print('\n[Original Image Metrics Test]')
    print(f'IoU: {Image_Result['IoU'].tolist()}')
    print(f'mIoU: {Image_Result['mIoU'].item()}')

    assert torch.allclose(
        Image_Result['IoU'],
        torch.ones(
            NUM_CLASSES,
            device=Device
        )
    )

    assert torch.isclose(
        Image_Result['mIoU'],
        torch.tensor(1.0, device=Device)
    )

    print('[PASS] Original Image Metric Test')

    # Validatioin Average Test
    Image_mIoU_List = [
        torch.tensor(1.0),
        torch.tensor(0.5),
        torch.tensor(0.0)
    ]

    Validation_mIoU = Calculate_Validation_mIoU(Image_mIoU_List)

    print('\n[Validation Average Test]')

    print(f'Validation mIoU: {Validation_mIoU.item()}')

    assert torch.isclose(Validation_mIoU, torch.tensor(0.5))

    print('[PASS] Validation Average Test')

    # Existing Metric Regression Test
    Prediction = torch.zeros(
        1,
        NUM_CLASSES,
        4,
        4,
        device=Device
    )

    Prediction[:, 0] = 1.0

    Target = torch.zeros(
        1,
        4,
        4,
        dtype=torch.long,
        device=Device
    )

    Existing_IoU = Calculate_IoU(Prediction, Target, NUM_CLASSES)

    Existing_mIoU = Calculate_mIoU(Prediction, Target, NUM_CLASSES)

    assert Existing_IoU.shape == (NUM_CLASSES,)
    assert torch.isfinite(Existing_mIoU)

    print('\n[Existing Metric Regression Test]')
    print('[PASS] Existing Metric Regression Test\n')

    print('=' * 60)
    print('FINAL RESULT: PASS')
    print('=' * 60)

# 4. Entry Point

if __name__ == '__main__':
    Main()