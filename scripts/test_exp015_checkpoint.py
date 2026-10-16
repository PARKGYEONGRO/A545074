from pathlib import Path

import torch

from src.boundary_model import Get_Boundary_Model
from scripts.train_boundary import (
    DEVICE,
    INIT_CHECKPOINT,
    Load_Initial_Checkpoint,
)


def Main():

    if not INIT_CHECKPOINT.exists():
        raise FileNotFoundError(
            f'Checkpoint not found: {INIT_CHECKPOINT}'
        )

    Model = Get_Boundary_Model().to(DEVICE)
    Model = Load_Initial_Checkpoint(Model, DEVICE)

    Model.eval()

    with torch.no_grad():
        Dummy_Input = torch.randn(1, 3, 384, 384, device=DEVICE)
        Output = Model(Dummy_Input)

    assert 'segmentation' in Output
    assert 'boundary' in Output

    assert Output['segmentation'].shape == (1, 4, 384, 384)
    assert Output['boundary'].shape == (1, 1, 384, 384)

    print('EXP-015 checkpoint test: PASS')
    print(f'Device: {DEVICE}')
    print(f"Segmentation output: {tuple(Output['segmentation'].shape)}")
    print(f"Boundary output: {tuple(Output['boundary'].shape)}")


if __name__ == '__main__':
    Main()