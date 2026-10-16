from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    TEST_IMAGE_DIR,
    PATCH_SIZE,
    NUM_CLASSES,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE
)

from src.patch_dataset import KoMaPPatchDataset
from src.transforms import Get_Valid_Transform
from src.boundary_model import Get_Boundary_Model
from src.evaluate import Get_Original_Image_Size


CHECKPOINT_PATH = (PROJECT_ROOT / 'outputs' / 'final' / 'exp011_final_model.pth')

OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'final' / 'test_predictions'


def Load_Final_Model(Device):

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f'Checkpoint not found: {CHECKPOINT_PATH}')

    Model = Get_Boundary_Model().to(Device)

    Checkpoint = torch.load(CHECKPOINT_PATH, map_location=Device, weights_only=False)

    if 'model_state_dict' not in Checkpoint:
        raise KeyError('model_state_dict not found in checkpoint')

    Model.load_state_dict(Checkpoint['model_state_dict'], strict=True)
    Model.eval()

    return Model


def Main():

    Device = torch.device(DEVICE)

    Test_Dataset = KoMaPPatchDataset(
        Image_Dir=TEST_IMAGE_DIR,
        Mask_Dir=None,
        Transform=Get_Valid_Transform(),
        Is_Test=True
    )

    Test_Dataloader = DataLoader(
        Test_Dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(Device.type == 'cuda')
    )

    Model = Load_Final_Model(Device)

    Probability_Sums = {}
    Prediction_Counts = {}
    Image_Paths = Test_Dataset.Base_Dataset.Image_Paths

    with torch.inference_mode():

        for Batch in Test_Dataloader:

            Images = Batch['image'].to(Device, non_blocking=True)

            Outputs = Model(Images)
            Logits = Outputs['segmentation']

            Probabilities = torch.softmax(Logits, dim=1).float().cpu()

            for Index, Image_Path in enumerate(Batch['image_path']):

                Image_Path = str(Image_Path)

                X = int(Batch['x'][Index].item())
                Y = int(Batch['y'][Index].item())

                Height, Width = Get_Original_Image_Size(
                    Image_Path
                )

                if Image_Path not in Probability_Sums:

                    Probability_Sums[Image_Path] = torch.zeros((NUM_CLASSES, Height, Width), dtype=torch.float32)

                    Prediction_Counts[Image_Path] = torch.zeros((Height, Width), dtype=torch.float32)

                Patch_Probabilities = Probabilities[Index]

                End_Y = min(Y + PATCH_SIZE, Height)
                End_X = min(X + PATCH_SIZE, Width)

                Actual_Height = End_Y - Y
                Actual_Width = End_X - X

                if Actual_Height <= 0 or Actual_Width <= 0:
                    raise ValueError(
                        f'Invalid patch coordinates: {Image_Path}, '
                        f'X={X}, Y={Y}'
                    )

                Probability_Sums[Image_Path][
                    :, Y:End_Y, X:End_X
                ] += Patch_Probabilities[
                    :, :Actual_Height, :Actual_Width
                ]

                Prediction_Counts[Image_Path][
                    Y:End_Y, X:End_X
                ] += 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    Expected_Paths = {
        str(Path(Path_Item))
        for Path_Item in Image_Paths
    }

    Actual_Paths = set(Probability_Sums)

    if Expected_Paths != Actual_Paths:
        raise RuntimeError(
            'Test image reconstruction mismatch. '
            f'Missing={Expected_Paths - Actual_Paths}; '
            f'Unexpected={Actual_Paths - Expected_Paths}'
        )

    Summary = []

    for Image_Path in sorted(Probability_Sums):

        Count_Map = Prediction_Counts[Image_Path]

        if torch.any(Count_Map == 0):
            raise RuntimeError(f'Uncovered pixels detected: {Image_Path}')

        Average_Probabilities = (Probability_Sums[Image_Path] / Count_Map.unsqueeze(0))

        Prediction_Mask = torch.argmax(Average_Probabilities, dim=0).numpy().astype(np.uint8)

        Height, Width = Get_Original_Image_Size(Image_Path)

        if Prediction_Mask.shape != (Height, Width):
            raise RuntimeError(f'Output size mismatch: {Image_Path}')

        Unique_Classes = np.unique(Prediction_Mask)

        if not np.isin(Unique_Classes, np.arange(NUM_CLASSES)).all():
            raise RuntimeError(
                f'Invalid class IDs: {Image_Path}, '
                f'{Unique_Classes}'
            )

        Image_Name = Path(Image_Path).name.replace('_image', '_mask')

        Output_Path = OUTPUT_DIR / Image_Name

        Image.fromarray(Prediction_Mask).save(Output_Path)

        Summary.append(
            (
                Path(Image_Path).name,
                Prediction_Mask.shape,
                Unique_Classes.tolist(),
                Output_Path.name
            )
        )

    Saved_Masks = list(OUTPUT_DIR.glob('*_mask.png'))

    if len(Saved_Masks) != len(Image_Paths):
        raise RuntimeError(
            f'Output count mismatch: expected={len(Image_Paths)}, '
            f'actual={len(Saved_Masks)}'
        )

    print('=' * 60)
    print('EXP-011 Test Inference Summary')
    print('=' * 60)

    for Image_Name, Shape, Classes, Output_Name in Summary:
        print(
            f'{Image_Name} -> {Output_Name} | '
            f'shape={Shape} | classes={Classes}'
        )

    print('-' * 60)
    print(f'Test images: {len(Image_Paths)}')
    print(f'Predictions: {len(Saved_Masks)}')
    print(f'Output directory: {OUTPUT_DIR}')
    print('Validation: PASS')


if __name__ == '__main__':
    Main()