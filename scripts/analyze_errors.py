# analyze_errors.py 검증 데이터로 평가 및 오분류를 수치로 비교
import csv
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

from src.config import DEVICE, NUM_CLASSES, CLASS_NAMES
from src.model import Get_Model
from src.evaluate import Evaluate_Model, Get_Validation_Dataloader


# 1. Analysis Settings
CHECKPOINT_PATHS = {
    'exp004': PROJECT_ROOT / 'outputs' / 'checkpoints' / 'best_model.pth',
    'exp006': PROJECT_ROOT / 'outputs' / 'checkpoints' / 'exp006' / 'best_model.pth',
}

TARGET_IMAGES = [
    'A71_step2_sample1_image.png',
    'A74_step3_sample4_image.png',
    'A153_step3_sample3_image.png',
]

OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'logs' / 'error_analysis'


# 2. Load Checkpoint
def Load_Checkpoint_Model(Checkpoint_Path, Device):
    if not Checkpoint_Path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {Checkpoint_Path}')

    Checkpoint = torch.load(
        Checkpoint_Path,
        map_location=Device,
        weights_only=False,
    )

    if not isinstance(Checkpoint, dict):
        raise TypeError(f'Invalid checkpoint format: {Checkpoint_Path}')

    if 'model_state_dict' not in Checkpoint:
        raise KeyError(f'model_state_dict not found: {Checkpoint_Path}')

    Model = Get_Model().to(Device)
    Model.load_state_dict(Checkpoint['model_state_dict'])
    Model.eval()

    print(f'Loaded: {Checkpoint_Path}')
    return Model


# 3. Calculate Confusion Matrix
# Rows: Ground Truth, Columns: Prediction
def Calculate_Confusion_Matrix(Prediction_Mask, Target_Mask):
    Prediction_Mask = Prediction_Mask.detach().cpu().numpy().astype(np.int64)
    Target_Mask = Target_Mask.detach().cpu().numpy().astype(np.int64)

    if Prediction_Mask.shape != Target_Mask.shape:
        raise ValueError('Prediction and target shapes do not match.')

    if not np.isin(Prediction_Mask, range(NUM_CLASSES)).all():
        raise ValueError('Prediction contains invalid class IDs.')

    if not np.isin(Target_Mask, range(NUM_CLASSES)).all():
        raise ValueError('Target contains invalid class IDs.')

    Encoded_Labels = (
        Target_Mask.ravel() * NUM_CLASSES
        + Prediction_Mask.ravel()
    )

    Confusion_Matrix = np.bincount(
        Encoded_Labels,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)

    return Confusion_Matrix


# 4. Calculate Class IoU from Confusion Matrix
def Calculate_Class_IoU(Confusion_Matrix):
    Intersection = np.diag(Confusion_Matrix).astype(np.float64)

    Ground_Truth_Count = Confusion_Matrix.sum(axis=1)
    Prediction_Count = Confusion_Matrix.sum(axis=0)

    Union = Ground_Truth_Count + Prediction_Count - Intersection

    Class_IoU = np.full(NUM_CLASSES, np.nan, dtype=np.float64)

    Valid_Classes = Union > 0
    Class_IoU[Valid_Classes] = (
        Intersection[Valid_Classes] / Union[Valid_Classes]
    )

    return Class_IoU


# 5. Save Confusion Matrix CSV
def Save_Confusion_Matrix(Confusion_Matrix, Output_Path):
    Output_Path.parent.mkdir(parents=True, exist_ok=True)

    with Output_Path.open(
        'w',
        newline='',
        encoding='utf-8-sig',
    ) as File:
        Writer = csv.writer(File)

        Writer.writerow(['Ground Truth / Prediction'] + CLASS_NAMES)

        for Class_ID, Class_Name in enumerate(CLASS_NAMES):
            Writer.writerow(
                [Class_Name] + Confusion_Matrix[Class_ID].tolist()
            )


# 6. Save Class IoU CSV
def Save_Class_IoU(Results, Output_Path):
    Output_Path.parent.mkdir(parents=True, exist_ok=True)

    with Output_Path.open('w', newline='', encoding='utf-8-sig',) as File:
        Writer = csv.writer(File)

        Writer.writerow(['Image', 'Model'] + CLASS_NAMES + ['mIoU'])

        for Result in Results:
            Writer.writerow([
                Result['Image'],
                Result['Model'],
                *[
                    (
                        f'{Value:.6f}'
                        if np.isfinite(Value)
                        else 'NaN'
                    )
                    for Value in Result['Class_IoU']
                ],
                f"{Result['mIoU']:.6f}",
            ])


# 7. Main Analysis
def Analyze_Errors():
    Device = torch.device(DEVICE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    Valid_Dataset, _ = Get_Validation_Dataloader()

    Validation_Image_Paths = {
        Path(Image_Path).name: str(Image_Path)
        for Image_Path in Valid_Dataset.Base_Dataset.Image_Paths
    }

    Missing_Images = [
        Image_Name
        for Image_Name in TARGET_IMAGES
        if Image_Name not in Validation_Image_Paths
    ]

    if Missing_Images:
        raise FileNotFoundError(f'Images not found in validation set: {Missing_Images}')

    All_Results = []

    for Experiment, Checkpoint_Path in CHECKPOINT_PATHS.items():
        print('\n' + '=' * 70)
        print(f'Analyzing {Experiment}')
        print('=' * 70)

        Model = Load_Checkpoint_Model(
            Checkpoint_Path=Checkpoint_Path,
            Device=Device,
        )

        Evaluation_Results = Evaluate_Model(
            Model=Model,
            Device=Device,
        )

        Predictions = Evaluation_Results['Predictions']

        Experiment_Confusion = np.zeros(
            (NUM_CLASSES, NUM_CLASSES),
            dtype=np.int64,
        )

        for Image_Name in TARGET_IMAGES:
            Image_Path = Validation_Image_Paths[Image_Name]
            Prediction_Mask = Predictions[Image_Path]
            Target_Mask = Valid_Dataset.Get_Original_Mask(Image_Path)

            Confusion_Matrix = Calculate_Confusion_Matrix(
                Prediction_Mask=Prediction_Mask,
                Target_Mask=Target_Mask,
            )

            Class_IoU = Calculate_Class_IoU(Confusion_Matrix)

            Valid_IoU = Class_IoU[np.isfinite(Class_IoU)]
            Image_mIoU = (
                float(Valid_IoU.mean())
                if len(Valid_IoU) > 0
                else 0.0
            )

            Experiment_Confusion += Confusion_Matrix

            All_Results.append({
                'Image': Image_Name,
                'Model': Experiment,
                'Class_IoU': Class_IoU,
                'mIoU': Image_mIoU,
            })

            print(f'\nImage: {Image_Name}')
            print(f'Image mIoU: {Image_mIoU:.6f}')

            print('\nConfusion Matrix')
            print('Rows = Ground Truth | Columns = Prediction')
            print('Classes:', CLASS_NAMES)
            print(Confusion_Matrix)

            print('\nClass IoU')
            for Class_Name, IoU_Value in zip(CLASS_NAMES, Class_IoU):
                print(f'{Class_Name}: {IoU_Value:.6f}')

            Save_Confusion_Matrix(
                Confusion_Matrix,
                OUTPUT_DIR / Experiment / f'{Path(Image_Name).stem}_confusion.csv',
            )

        Save_Confusion_Matrix(
            Experiment_Confusion,
            OUTPUT_DIR / Experiment / 'combined_confusion.csv',
        )

        print('\nCombined Confusion Matrix')
        print(Experiment_Confusion)

        del Model
        if Device.type == 'cuda':
            torch.cuda.empty_cache()

    Save_Class_IoU(
        All_Results,
        OUTPUT_DIR / 'class_iou_comparison.csv',
    )

    print('\n' + '=' * 70)
    print('Error analysis completed.')
    print(f'Output directory: {OUTPUT_DIR}')
    print('=' * 70)


if __name__ == '__main__':
    Analyze_Errors()
