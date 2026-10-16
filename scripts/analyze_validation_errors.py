# analyze_validation_errors.py 검증 이미지 20장의 예측을 원본 해상도로 복원 및 분석
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
    'exp007': PROJECT_ROOT / 'outputs' / 'checkpoints' / 'exp007' / 'best_model.pth',
}

OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'logs' / 'validation_error_analysis_exp007_comparison'


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
        raise TypeError(f'Invalid checkpoint: {Checkpoint_Path}')

    if 'model_state_dict' not in Checkpoint:
        raise KeyError(f'model_state_dict not found: {Checkpoint_Path}')

    Model = Get_Model().to(Device)
    Model.load_state_dict(Checkpoint['model_state_dict'])
    Model.eval()

    return Model


# 3. Calculate Confusion Matrix
# Row: Ground Truth, Columns: Predictiion
def Calculate_Confusion_Matrix(Prediction_Mask, Target_Mask):
    Prediction_Mask = (Prediction_Mask.detach().cpu().numpy().astype(np.int64))
    Target_Mask = (Target_Mask.detach().cpu().numpy().astype(np.int64))

    if Prediction_Mask.shape != Target_Mask.shape:
        raise ValueError('Prediction and target shapes do not match.')

    if not np.isin(Prediction_Mask, range(NUM_CLASSES)).all():
        raise ValueError('Invalid prediction class ID.')

    if not np.isin(Target_Mask, range(NUM_CLASSES)).all():
        raise ValueError('Invalid target class ID.')

    Encoded_Labels = (
        Target_Mask.ravel() * NUM_CLASSES
        + Prediction_Mask.ravel()
    )

    return np.bincount(
        Encoded_Labels,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)


# 4. Calculate Metrics
def Calculate_Confusion_Metrics(Confusion_Matrix):
    Intersection = np.diag(Confusion_Matrix).astype(np.float64)
    Target_Count = Confusion_Matrix.sum(axis=1)
    Prediction_Count = Confusion_Matrix.sum(axis=0)
    Union = Target_Count + Prediction_Count - Intersection

    Class_IoU = np.full(NUM_CLASSES, np.nan, dtype=np.float64)

    Valid_Classes = Union > 0
    Class_IoU[Valid_Classes] = (Intersection[Valid_Classes] / Union[Valid_Classes])

    Target_Fraction = Target_Count / max(Target_Count.sum(), 1)
    Prediction_Fraction = (Prediction_Count / max(Prediction_Count.sum(), 1))

    return {
        'Class_IoU': Class_IoU,
        'Target_Count': Target_Count,
        'Prediction_Count': Prediction_Count,
        'Target_Fraction': Target_Fraction,
        'Prediction_Fraction': Prediction_Fraction,
    }


# 5. Save Confusion Matrix
def Save_Confusion_Matrix(Confusion_Matrix, Output_Path):
    Output_Path.parent.mkdir(parents=True, exist_ok=True)

    with Output_Path.open('w', newline='', encoding='utf-8-sig',) as File:
        Writer = csv.writer(File)
        Writer.writerow(['Ground Truth / Prediction'] + CLASS_NAMES)

        for Class_ID, Class_Name in enumerate(CLASS_NAMES):
            Writer.writerow([Class_Name] + Confusion_Matrix[Class_ID].tolist())


# 6. Save Per-Class Summary
def Save_Class_Summary(Summary_Rows, Output_Path):
    Output_Path.parent.mkdir(parents=True, exist_ok=True)

    with Output_Path.open('w', newline='', encoding='utf-8-sig',) as File:
        Writer = csv.writer(File)

        Writer.writerow([
            'Model',
            'Class',
            'Target_Pixels',
            'Predicted_Pixels',
            'Target_Fraction',
            'Prediction_Fraction',
            'Global_Pixel_IoU',
            'Mean_Per_Image_IoU',
        ])

        Writer.writerows(Summary_Rows)


# 7. Main Analysis
def Analyze_Validation_Errors():
    Device = torch.device(DEVICE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    Valid_Dataset, _ = Get_Validation_Dataloader()

    Image_Paths = {
        str(Path(Image_Path)): str(Image_Path)
        for Image_Path in Valid_Dataset.Base_Dataset.Image_Paths
    }

    Summary_Rows = []
    Model_Results = {}

    for Experiment, Checkpoint_Path in CHECKPOINT_PATHS.items():
        print('\n' + '=' * 75)
        print(f'Analyzing {Experiment}')
        print('=' * 75)

        Model = Load_Checkpoint_Model(Checkpoint_Path, Device)

        Results = Evaluate_Model(Model=Model, Device=Device)
        Predictions = Results['Predictions']

        if len(Predictions) != len(Image_Paths):
            raise RuntimeError(f'Expected {len(Image_Paths)} predictions, got {len(Predictions)}.')

        Total_Confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64,)

        Per_Image_IoU = []

        for Image_Path, Prediction_Mask in Predictions.items():
            Image_Key = str(Path(Image_Path))

            if Image_Key not in Image_Paths:
                raise KeyError(f'Unexpected validation image: {Image_Path}')

            Target_Mask = Valid_Dataset.Get_Original_Mask(Image_Path)

            Confusion_Matrix = Calculate_Confusion_Matrix(Prediction_Mask, Target_Mask,)

            Metrics = Calculate_Confusion_Metrics(Confusion_Matrix)

            Total_Confusion += Confusion_Matrix
            Per_Image_IoU.append(Metrics['Class_IoU'])

            Image_Name = Path(Image_Path).name

            Save_Confusion_Matrix(
                Confusion_Matrix,
                OUTPUT_DIR / Experiment / 'per_image' / f'{Path(Image_Name).stem}_confusion.csv',
            )

        Global_Metrics = Calculate_Confusion_Metrics(Total_Confusion)

        Per_Image_IoU = np.asarray(Per_Image_IoU, dtype=np.float64)

        Mean_Per_Image_IoU = np.full(
            NUM_CLASSES,
            np.nan,
            dtype=np.float64,
        )

        for Class_ID in range(NUM_CLASSES):
            Valid_IoU = Per_Image_IoU[:, Class_ID]
            Valid_IoU = Valid_IoU[np.isfinite(Valid_IoU)]

            if len(Valid_IoU) > 0:
                Mean_Per_Image_IoU[Class_ID] = Valid_IoU.mean()

        Save_Confusion_Matrix(
            Total_Confusion,
            OUTPUT_DIR / Experiment / 'combined_confusion.csv',
        )

        print('\nCombined Confusion Matrix')
        print('Rows = Ground Truth, Columns = Prediction')
        print(CLASS_NAMES)
        print(Total_Confusion)

        print('\nClass Summary')
        print(
            f'{"Class":<18} {"Global IoU":>12} '
            f'{"Mean Image IoU":>16} {"Target %":>12} {"Predicted %":>12}'
        )

        for Class_ID, Class_Name in enumerate(CLASS_NAMES):
            Global_IoU = Global_Metrics['Class_IoU'][Class_ID]
            Mean_IoU = Mean_Per_Image_IoU[Class_ID]

            print(
                f'{Class_Name:<18} '
                f'{Global_IoU:>12.6f} '
                f'{Mean_IoU:>16.6f} '
                f'{Global_Metrics["Target_Fraction"][Class_ID] * 100:>11.3f}% '
                f'{Global_Metrics["Prediction_Fraction"][Class_ID] * 100:>11.3f}%'
            )

            Summary_Rows.append([
                Experiment,
                Class_Name,
                int(Global_Metrics['Target_Count'][Class_ID]),
                int(Global_Metrics['Prediction_Count'][Class_ID]),
                f'{Global_Metrics["Target_Fraction"][Class_ID]:.8f}',
                f'{Global_Metrics["Prediction_Fraction"][Class_ID]:.8f}',
                (
                    f'{Global_IoU:.8f}'
                    if np.isfinite(Global_IoU)
                    else 'NaN'
                ),
                (
                    f'{Mean_IoU:.8f}'
                    if np.isfinite(Mean_IoU)
                    else 'NaN'
                ),
            ])

        Model_Results[Experiment] = {
            'Confusion': Total_Confusion,
            'Metrics': Global_Metrics,
            'Mean_Per_Image_IoU': Mean_Per_Image_IoU,
        }

        del Model

        if Device.type == 'cuda':
            torch.cuda.empty_cache()

    Save_Class_Summary(
        Summary_Rows,
        OUTPUT_DIR / 'class_summary.csv',
    )


    # Compare Experiments
    Comparison_Pairs = [
        ('exp004', 'exp006'),
        ('exp006', 'exp007'),
    ]

    for Baseline_Name, Compare_Name in Comparison_Pairs:
        if (
            Baseline_Name not in Model_Results
            or Compare_Name not in Model_Results
        ):
            continue

        print('\n' + '=' * 85)
        print(f'{Compare_Name.upper()} minus {Baseline_Name.upper()}')
        print('=' * 85)

        Baseline = Model_Results[Baseline_Name]
        Compare = Model_Results[Compare_Name]

        print(
            f'{"Class":<18} '
            f'{"Global IoU Delta":>17} '
            f'{"Mean Image IoU Delta":>22} '
            f'{"Prediction % Delta":>20}'
        )

        for Class_ID, Class_Name in enumerate(CLASS_NAMES):
            Baseline_IoU = Baseline['Metrics']['Class_IoU'][Class_ID]
            Compare_IoU = Compare['Metrics']['Class_IoU'][Class_ID]

            Baseline_Mean = Baseline['Mean_Per_Image_IoU'][Class_ID]
            Compare_Mean = Compare['Mean_Per_Image_IoU'][Class_ID]

            Baseline_Fraction = (Baseline['Metrics']['Prediction_Fraction'][Class_ID])
            Compare_Fraction = (Compare['Metrics']['Prediction_Fraction'][Class_ID])

            Global_Delta = (
                Compare_IoU - Baseline_IoU
                if np.isfinite(Baseline_IoU)
                and np.isfinite(Compare_IoU)
                else np.nan
            )

            Mean_Delta = (
                Compare_Mean - Baseline_Mean
                if np.isfinite(Baseline_Mean)
                and np.isfinite(Compare_Mean)
                else np.nan
            )

            Prediction_Delta = (Compare_Fraction - Baseline_Fraction) * 100

            print(
                f'{Class_Name:<18} '
                f'{Global_Delta:>17.6f} '
                f'{Mean_Delta:>22.6f} '
                f'{Prediction_Delta:>19.3f}%'
            )

    print('\nAnalysis completed.')
    print(f'Results saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    Analyze_Validation_Errors()