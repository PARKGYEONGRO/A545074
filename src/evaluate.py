# evaluate.py

from pathlib import Path

import torch
import sys

from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)

from src.config import(
    VALID_IMAGE_DIR,
    VALID_MASK_DIR,
    NUM_CLASSES,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE
)

from src.patch_dataset import KoMaPPatchDataset
from src.transforms import Get_Valid_Transform
from src.model import Get_Model

from src.metrics import(
    Calculate_Image_Metrics,
    Calculate_Validation_mIoU
)


# 1. Validation Dataset
def Get_Validation_Dataloader():
    Valid_Dataset = KoMaPPatchDataset(
        Image_Dir=VALID_IMAGE_DIR,
        Mask_Dir=VALID_MASK_DIR,
        Transform=Get_Valid_Transform(),
        Is_Test=False
    )

    Valid_Dataloader = DataLoader(
        Valid_Dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == 'cuda')
    )

    return Valid_Dataset, Valid_Dataloader


# 2. Original Image Size
def Get_Original_Image_Size(Image_Path):
    with Image.open(Image_Path) as Image_File:
        Width, Height = Image_File.size
    return Height, Width


# 3. Original Image Prediction Reconstruction
def Reconstruct_Original_Predictions(Model, Valid_Dataset, Valid_Dataloader, Device):
    Model.eval()

    # 3-1. Image Accumulator
    Prediction_Sums = {}
    Prediction_Counts = {}

    with torch.inference_mode():
        for Batch in Valid_Dataloader:
            Images = Batch['image'].to(Device, non_blocking=True)

            Image_Paths = Batch['image_path']
            Patch_Xs = Batch['x']
            Patch_Ys = Batch['y']

            # 3-2. Patch Prediction
            Logits = Model(Images)

            Probabilities = torch.softmax(Logits, dim=1)

            Probabilities = Probabilities.cpu()

            # 3-3. Accumulate Each Patch
            for Index, Image_Path in enumerate(Image_Paths):
                Image_Path = str(Image_Path)

                X = int(Patch_Xs[Index].item())
                Y = int(Patch_Ys[Index].item())

                Height, Width = Get_Original_Image_Size(Image_Path)

                if Image_Path not in Prediction_Sums:
                    Prediction_Sums[Image_Path] = torch.zeros(
                        (NUM_CLASSES, Height, Width),
                        dtype=torch.float32
                    )

                    Prediction_Counts[Image_Path] = torch.zeros(
                        (Height, Width),
                        dtype=torch.float32
                    )

                Patch_Probabilities = Probabilities[Index]
                Patch_Height = Patch_Probabilities.shape[1]
                Patch_Width = Patch_Probabilities.shape[2]

                # 3-4. Boundary-Safe Coordinates
                End_Y = min(Y + Patch_Height, Height)
                End_X = min(X + Patch_Width, Width)

                Actual_Height = End_Y - Y
                Actual_Width = End_X - X

                if Actual_Height <= 0 or Actual_Width <= 0:
                    raise ValueError(
                        f'Invalid patch coordinates: {Image_Path}, X= {X}, Y={Y}'
                    )

                # 3-5. Probability Accumulation
                Prediction_Sums[Image_Path][:, Y:End_Y, X:End_X] += Patch_Probabilities[:, :Actual_Height, :Actual_Width]
                Prediction_Counts[Image_Path][Y:End_Y, X:End_X] += 1


        # 3-6 Average Probabilities and Decode Classes
        Original_Predictions = {}

        for Image_Path, Probability_Sum in Prediction_Sums.items():
            Count_Map = Prediction_Counts[Image_Path]

            if torch.any(Count_Map == 0):
                raise RuntimeError(f'Uncovered pixels detected: {Image_Path}')

            Average_Probabilities = (Probability_Sum / Count_Map.unsqueeze(0))

            Prediction_Mask = torch.argmax(Average_Probabilities, dim=0)

            Original_Predictions[Image_Path] = Prediction_Mask

        Expected_Image_Count = len(Valid_Dataset.Base_Dataset.Image_Paths)

        if len(Original_Predictions) != Expected_Image_Count:
            raise RuntimeError('Validation image reconstruction count mismatch:'
                               f'Expected={Expected_Image_Count}, Actual={len(Original_Predictions)}')

        return Original_Predictions

# 4. Image-Level Validation Metrics
def Evaluate_Original_Images(Valid_Dataset, Original_Predictions):
    Image_Metrics = {}
    Image_mIoU_List = []

    for Image_Path, Prediction_Mask in Original_Predictions.items():
        Target_Mask = Valid_Dataset.Get_Original_Mask(Image_Path)

        if Prediction_Mask.shape != Target_Mask.shape:
            raise ValueError(f'Prediction and target size mismatch: {Image_Path}')

        Metrics = Calculate_Image_Metrics(
            Prediction_Mask=Prediction_Mask,
            Target_Mask=Target_Mask,
            Num_Classes=NUM_CLASSES
        )

        Image_Metrics[Image_Path] = Metrics

        Image_mIoU_List.append(Metrics['mIoU'].detach().cpu())

    Validation_mIoU = Calculate_Validation_mIoU(Image_mIoU_List)

    return Validation_mIoU, Image_Metrics

# 5. Full Validation
def Evaluate_Model(Model, Device=None):

    if Device is None:
        Device = torch.device(DEVICE)

    else:
        Device = torch.device(Device)

    Valid_Dataset, Valid_Dataloader = (Get_Validation_Dataloader())

    Model = Model.to(Device)

    Original_Predictions = (
        Reconstruct_Original_Predictions(
            Model=Model,
            Valid_Dataset=Valid_Dataset,
            Valid_Dataloader=Valid_Dataloader,
            Device=Device
        )
    )

    Validation_mIoU, Image_Metrics = (
        Evaluate_Original_Images(
            Valid_Dataset=Valid_Dataset,
            Original_Predictions=Original_Predictions
        )
    )

    print('=' * 60)
    print('KoMaP Original Image Validation')
    print('=' * 60)

    for Image_Path, Metrics in Image_Metrics.items():
        
        Image_Name = Path(Image_Path).name

        IoU_Values = Metrics['IoU'].tolist()
        Image_mIoU = Metrics['mIoU'].item()

        print(f'Image: {Image_Name}')
        print(f'Class IoU: {IoU_Values}')
        print(f'Image mIoU: {Image_mIoU:.6f}')
        print('-' * 60)

    print(f'Validation mIoU: {Validation_mIoU.item():.6f}')

    return {
        'Validation_mIoU': Validation_mIoU,
        'Image_Metrics': Image_Metrics,
        'Predictions': Original_Predictions
    }

# 6. Standalone Entry Point
if __name__ == '__main__':
    Model = Get_Model()

    Results = Evaluate_Model(Model)

    print('\nEvaluation completed.\nNote: This model has not been trained yet.')