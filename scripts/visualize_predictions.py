# scripts/visualize_predictions.py 이미지 시각화
from pathlib import Path
import sys
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)

from src.config import DEVICE, NUM_CLASSES
from src.model import Get_Model
from src.boundary_model import Get_Boundary_Model
from src.evaluate import(
    Get_Validation_Dataloader,
    Reconstruct_Original_Predictions
)

class SegmentationOnlyModel(torch.nn.Module):
    def __init__(self, Model):
        super().__init__()
        self.Model = Model

    def forward(self, Images):
        Outputs = self.Model(Images)
        return Outputs['segmentation']


# 1. Class configuration
CLASS_NAMES = [
    'Al',
    'Al3Ni',
    'Primary_Si',
    'Eutectic_Si'
]

CLASS_COLORS = np.array([
    [153, 127, 76],
    [25, 76, 153],
    [76, 178, 76],
    [204, 204, 204]
], dtype=np.uint8)


# 2. Mask Color Conversion
def Convert_Mask_To_RGB(Mask):
    Mask_Numpy = (
        Mask.detach().cpu().numpy()
        if isinstance(Mask, torch.Tensor)
        else np.asarray(Mask)
    )

    if Mask_Numpy.ndim != 2:
        raise ValueError(f'Expected 2D class mask, got {Mask_Numpy.shape}')

    if (Mask_Numpy.min() < 0 or Mask_Numpy.max() >= NUM_CLASSES):
        raise ValueError('Mask contains invalid class IDs.')

    return CLASS_COLORS[Mask_Numpy]


# 3. Visualization
def Save_Visualization(Image_Path, Target_Mask, Prediction_Mask, Output_Dir):
    Image_Path = Path(Image_Path)

    with Image.open(Image_Path) as Image_File:
        Original_Image = np.array(
            Image_File.convert('RGB')
        )

    Target_RGB = Convert_Mask_To_RGB(Target_Mask)

    Prediction_RGB = Convert_Mask_To_RGB(Prediction_Mask)

    if (
        Original_Image.shape[:2] != Target_RGB.shape[:2]
        or Target_RGB.shape != Prediction_RGB.shape
    ):
        raise ValueError(f'Visualization size mismatch: {Image_Path.name}')

    Overlay = (
        Original_Image.astype(np.float32) * 0.5
        + Prediction_RGB.astype(np.float32) * 0.5
    ).clip(0, 255).astype(np.uint8)

    Figure, Axes = plt.subplots(
        1,
        4,
        figsize=(20, 6)
    )

    Axes[0].imshow(Original_Image)
    Axes[0].set_title('Original Image')

    Axes[1].imshow(Target_RGB)
    Axes[1].set_title('Ground Truth')

    Axes[2].imshow(Prediction_RGB)
    Axes[2].set_title('Prediction')

    Axes[3].imshow(Overlay)
    Axes[3].set_title('Prediction Overlay')

    for Axis in Axes:
        Axis.axis('off')

    Figure.suptitle(Image_Path.name, fontsize=12)
    Figure.tight_layout()

    Output_Dir.mkdir(parents=True, exist_ok=True)
    Output_Path = (Output_Dir / f'{Image_Path.stem}_visualization.png')

    Figure.savefig(Output_Path, dpi=150, bbox_inches='tight')

    plt.close(Figure)

    return Output_Path


# 4. Save Model Comparison Visualization
def Save_Comparison_Visualization(
    Image_Path,
    Target_Mask,
    Prediction_009,
    Prediction_011_boundary,
    Output_Dir
):
    Image_Path = Path(Image_Path)

    with Image.open(Image_Path) as Image_File:
        Original_Image = np.array(Image_File.convert('RGB'))

    Target = (
        Target_Mask.detach().cpu().numpy()
        if isinstance(Target_Mask, torch.Tensor)
        else np.asarray(Target_Mask)
    )

    Prediction_009 = (
        Prediction_009.detach().cpu().numpy()
        if isinstance(Prediction_009, torch.Tensor)
        else np.asarray(Prediction_009)
    )

    Prediction_011_boundary = (
        Prediction_011_boundary.detach().cpu().numpy()
        if isinstance(Prediction_011_boundary, torch.Tensor)
        else np.asarray(Prediction_011_boundary)
    )

    if not (
        Target.shape == Prediction_009.shape
        == Prediction_011_boundary.shape
        == Original_Image.shape[:2]
    ):
        raise ValueError(f'Comparison size mismatch: {Image_Path.name}')

    Correct_009 = Prediction_009 == Target
    Correct_011_boundary = Prediction_011_boundary == Target

    Improved = (~Correct_009) & Correct_011_boundary
    Regressed = Correct_009 & (~Correct_011_boundary)

    Both_Wrong = (
        (~Correct_009)
        & (~Correct_011_boundary)
        & (Prediction_009 != Prediction_011_boundary)
    )

    Difference_RGB = np.zeros((*Target.shape, 3), dtype=np.uint8)

    # Green: EXP-011_boundary corrected an EXP-009 error
    Difference_RGB[Improved] = [0, 200, 0]

    # Red: EXP-011_boundary introduced an error
    Difference_RGB[Regressed] = [230, 30, 30]

    # Yellow: Both models are wrong and predict different classes
    Difference_RGB[Both_Wrong] = [255, 220, 0]

    Target_RGB = Convert_Mask_To_RGB(Target)
    Prediction_009_RGB = Convert_Mask_To_RGB(Prediction_009)
    Prediction_011_boundary_RGB = Convert_Mask_To_RGB(Prediction_011_boundary)

    Overlay_011_boundary = (
        Original_Image.astype(np.float32) * 0.5
        + Prediction_011_boundary_RGB.astype(np.float32) * 0.5
    ).clip(0, 255).astype(np.uint8)

    Figure, Axes = plt.subplots(2, 3, figsize=(20, 12))

    Panels = [
        (Original_Image, 'Original Image'),
        (Target_RGB, 'Ground Truth'),
        (Prediction_009_RGB, 'EXP-009 Prediction'),
        (Prediction_011_boundary_RGB, 'EXP-011_boundary Prediction'),
        (Difference_RGB, 'Prediction Difference'),
        (Overlay_011_boundary, 'EXP-011_boundary Overlay'),
    ]

    for Axis, (Panel_Image, Panel_Title) in zip(Axes.ravel(), Panels):
        Axis.imshow(Panel_Image)
        Axis.set_title(Panel_Title)
        Axis.axis('off')

    Figure.suptitle(f'{Image_Path.name}\nGreen: Improved | Red: Regressed | Yellow: Both Wrong', fontsize=14)

    Figure.tight_layout()

    Output_Dir.mkdir(parents=True, exist_ok=True)

    Output_Path = (Output_Dir / f'{Image_Path.stem}_comparison.png')

    Figure.savefig(Output_Path, dpi=150, bbox_inches='tight')

    plt.close(Figure)

    # Pixel-level comparison statistics
    Total_Pixels = Target.size

    print(f'\nComparison: {Image_Path.name}')
    print(f'Total Pixels: {Total_Pixels:,}')
    print(
        f'EXP-011_boundary Improved Pixels: '
        f'{int(Improved.sum()):,} '
        f'({Improved.mean() * 100:.4f}%)'
    )
    print(
        f'EXP-011_boundary Regressed Pixels: '
        f'{int(Regressed.sum()):,} '
        f'({Regressed.mean() * 100:.4f}%)'
    )
    print(
        f'Both Wrong, Different Prediction: '
        f'{int(Both_Wrong.sum()):,} '
        f'({Both_Wrong.mean() * 100:.4f}%)'
    )

    print('\nPer-Class Improved / Regressed Pixels')

    for Class_ID, Class_Name in enumerate(CLASS_NAMES):
        Class_Pixels = Target == Class_ID

        Improved_Count = int((Improved & Class_Pixels).sum())

        Regressed_Count = int((Regressed & Class_Pixels).sum())

        print(
            f'{Class_Name:<14} '
            f'Improved={Improved_Count:>9,} '
            f'Regressed={Regressed_Count:>9,}'
        )

    return Output_Path


# 5. Print_Class_Distribution
def Print_Class_Distribution(Mask, Title):
    Mask_Numpy = (
        Mask.detach().cpu().numpy()
        if isinstance(Mask, torch.Tensor)
        else np.asarray(Mask)
    )

    Pixel_Counts = np.bincount(Mask_Numpy.astype(np.int64).ravel(), minlength=NUM_CLASSES)

    Total_Pixels = Pixel_Counts.sum()

    print(f'\n{Title}')

    for Class_ID, Class_Name in enumerate(CLASS_NAMES):
        Pixel_Ratio = (
            Pixel_Counts[Class_ID] / Total_Pixels * 100
            if Total_Pixels > 0
            else 0.0
        )

        print(
            f'{Class_Name:<12}'
            f'ID={Class_ID}'
            f'Pixels={Pixel_Counts[Class_ID]:>9,}'
            f'Ratio={Pixel_Ratio:>7.3f}%'
        )


# 6. Analyze Class Transition Errors
def Analyze_Class_Transitions(Valid_Dataset, Predictions_009, Predictions_011_boundary, Output_Dir):
    import csv

    Output_Dir.mkdir(
        parents=True,
        exist_ok=True
    )

    def Normalize_Path(Image_Path):
        return str(Path(Image_Path).resolve()).casefold()

    Predictions_009_By_Path = {
        Normalize_Path(Image_Path): Prediction
        for Image_Path, Prediction in Predictions_009.items()
    }

    Predictions_011_boundary_By_Path = {
        Normalize_Path(Image_Path): Prediction
        for Image_Path, Prediction in Predictions_011_boundary.items()
    }

    if set(Predictions_009_By_Path) != set(Predictions_011_boundary_By_Path):
        raise RuntimeError(
            'EXP-009 and EXP-011_boundary prediction image sets do not match.'
        )

    # Dimensions: Ground Truth x EXP-009 x EXP-011_boundary
    Transition_Counts = np.zeros(
        (NUM_CLASSES, NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    # Class-level IoU statistics
    Confusion_009 = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    Confusion_011_boundary = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    Per_Image_IoU_Sum_009 = np.zeros(
        NUM_CLASSES,
        dtype=np.float64
    )

    Per_Image_IoU_Sum_011_boundary = np.zeros(
        NUM_CLASSES,
        dtype=np.float64
    )

    Per_Image_IoU_Count_009 = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    Per_Image_IoU_Count_011_boundary = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )


    Image_Summary_Rows = []

    Per_Class_Counts = {
        Class_ID: {
            'Improved': 0,
            'Regressed': 0,
            'Both_Correct': 0,
            'Both_Wrong': 0,
            'Same_Prediction': 0,
        }
        for Class_ID in range(NUM_CLASSES)
    }

    Image_Paths = Valid_Dataset.Base_Dataset.Image_Paths

    for Image_Path in Image_Paths:
        Image_Path = Path(Image_Path)
        Image_Key = Normalize_Path(Image_Path)

        if Image_Key not in Predictions_009_By_Path:
            raise KeyError(
                f'Validation prediction missing: {Image_Path}'
            )

        Target_Mask = Valid_Dataset.Get_Original_Mask(
            str(Image_Path)
        )

        Target = (
            Target_Mask.detach().cpu().numpy()
            if isinstance(Target_Mask, torch.Tensor)
            else np.asarray(Target_Mask)
        ).astype(np.int64)

        Prediction_009 = (
            Predictions_009_By_Path[Image_Key]
            .detach().cpu().numpy()
            if isinstance(Predictions_009_By_Path[Image_Key], torch.Tensor)
            else np.asarray(Predictions_009_By_Path[Image_Key])
        ).astype(np.int64)

        Prediction_011_boundary = (
            Predictions_011_boundary_By_Path[Image_Key]
            .detach().cpu().numpy()
            if isinstance(Predictions_011_boundary_By_Path[Image_Key], torch.Tensor)
            else np.asarray(Predictions_011_boundary_By_Path[Image_Key])
        ).astype(np.int64)

        if not (
            Target.shape == Prediction_009.shape
            == Prediction_011_boundary.shape
        ):
            raise ValueError(
                f'Mask shape mismatch: {Image_Path.name}'
            )

        for Mask_Name, Mask in [
            ('Target', Target),
            ('EXP-009', Prediction_009),
            ('EXP-011_boundary', Prediction_011_boundary),
        ]:
            if not np.isin(Mask, range(NUM_CLASSES)).all():
                raise ValueError(
                    f'Invalid class ID in {Mask_Name}: {Image_Path.name}'
                )


            # Update global confusion matrices
        Confusion_009 += np.bincount(
            Target.ravel() * NUM_CLASSES + Prediction_009.ravel(),
            minlength=NUM_CLASSES ** 2
        ).reshape(NUM_CLASSES, NUM_CLASSES)

        Confusion_011_boundary += np.bincount(
            Target.ravel() * NUM_CLASSES + Prediction_011_boundary.ravel(),
            minlength=NUM_CLASSES ** 2
        ).reshape(NUM_CLASSES, NUM_CLASSES)


        # Calculate per-image class IoU
        for Class_ID in range(NUM_CLASSES):
            Target_Class = Target == Class_ID

            Prediction_Class_009 = Prediction_009 == Class_ID
            Prediction_Class_011_boundary = Prediction_011_boundary == Class_ID

            Intersection_009 = int(
                (Target_Class & Prediction_Class_009).sum()
            )

            Union_009 = int(
                (Target_Class | Prediction_Class_009).sum()
            )

            Intersection_011_boundary = int(
                (Target_Class & Prediction_Class_011_boundary).sum()
            )

            Union_011_boundary = int(
                (Target_Class | Prediction_Class_011_boundary).sum()
            )

            if Union_009 > 0:
                Per_Image_IoU_Sum_009[Class_ID] += (
                    Intersection_009 / Union_009
                )
                Per_Image_IoU_Count_009[Class_ID] += 1

            if Union_011_boundary > 0:
                Per_Image_IoU_Sum_011_boundary[Class_ID] += (
                    Intersection_011_boundary / Union_011_boundary
                )
                Per_Image_IoU_Count_011_boundary[Class_ID] += 1

        # Encode each pixel as:
        # GT * NUM_CLASSES^2 + EXP-009 * NUM_CLASSES + EXP-011_boundary
        Encoded = (
            Target.ravel() * NUM_CLASSES * NUM_CLASSES
            + Prediction_009.ravel() * NUM_CLASSES
            + Prediction_011_boundary.ravel()
        )

        Image_Transitions = np.bincount(
            Encoded,
            minlength=NUM_CLASSES ** 3
        ).reshape(
            NUM_CLASSES,
            NUM_CLASSES,
            NUM_CLASSES
        )

        Transition_Counts += Image_Transitions

        Correct_009 = Prediction_009 == Target
        Correct_011_boundary = Prediction_011_boundary == Target

        Improved = (~Correct_009) & Correct_011_boundary
        Regressed = Correct_009 & (~Correct_011_boundary)
        Both_Correct = Correct_009 & Correct_011_boundary
        Both_Wrong = (~Correct_009) & (~Correct_011_boundary)

        Image_Summary_Rows.append([
            Image_Path.name,
            int(Target.size),
            int(Improved.sum()),
            int(Regressed.sum()),
            int(Both_Correct.sum()),
            int(Both_Wrong.sum()),
            int((Prediction_009 == Prediction_011_boundary).sum()),
        ])

        for Class_ID in range(NUM_CLASSES):
            Class_Pixels = Target == Class_ID

            Per_Class_Counts[Class_ID]['Improved'] += int(
                (Improved & Class_Pixels).sum()
            )

            Per_Class_Counts[Class_ID]['Regressed'] += int(
                (Regressed & Class_Pixels).sum()
            )

            Per_Class_Counts[Class_ID]['Both_Correct'] += int(
                (Both_Correct & Class_Pixels).sum()
            )

            Per_Class_Counts[Class_ID]['Both_Wrong'] += int(
                (Both_Wrong & Class_Pixels).sum()
            )

            Per_Class_Counts[Class_ID]['Same_Prediction'] += int(
                (
                    (Prediction_009 == Prediction_011_boundary)
                    & Class_Pixels
                ).sum()
            )

        # Save per-image transition combinations
        Image_Output_Path = (
            Output_Dir
            / 'per_image'
            / f'{Image_Path.stem}_transitions.csv'
        )

        Image_Output_Path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with Image_Output_Path.open(
            'w',
            newline='',
            encoding='utf-8-sig'
        ) as File:
            Writer = csv.writer(File)

            Writer.writerow([
                'Ground_Truth',
                'EXP009_Prediction',
                'EXP011_boundary_Prediction',
                'Pixel_Count',
                'Outcome',
            ])

            for GT_ID in range(NUM_CLASSES):
                for Pred_009_ID in range(NUM_CLASSES):
                    for Pred_011_boundary_ID in range(NUM_CLASSES):
                        Count = int(
                            Image_Transitions[
                                GT_ID,
                                Pred_009_ID,
                                Pred_011_boundary_ID
                            ]
                        )

                        if Count == 0:
                            continue

                        Is_Correct_009 = GT_ID == Pred_009_ID
                        Is_Correct_011_boundary = GT_ID == Pred_011_boundary_ID

                        if not Is_Correct_009 and Is_Correct_011_boundary:
                            Outcome = 'Improved'
                        elif Is_Correct_009 and not Is_Correct_011_boundary:
                            Outcome = 'Regressed'
                        elif Is_Correct_009 and Is_Correct_011_boundary:
                            Outcome = 'Both_Correct'
                        elif Pred_009_ID == Pred_011_boundary_ID:
                            Outcome = 'Both_Wrong_Same_Prediction'
                        else:
                            Outcome = 'Both_Wrong_Different_Prediction'

                        Writer.writerow([
                            CLASS_NAMES[GT_ID],
                            CLASS_NAMES[Pred_009_ID],
                            CLASS_NAMES[Pred_011_boundary_ID],
                            Count,
                            Outcome,
                        ])

    # Save all-image transition combinations
    Combined_Path = Output_Dir / 'combined_transitions.csv'

    with Combined_Path.open(
        'w',
        newline='',
        encoding='utf-8-sig'
    ) as File:
        Writer = csv.writer(File)

        Writer.writerow([
            'Ground_Truth',
            'EXP009_Prediction',
            'EXP011_boundary_Prediction',
            'Pixel_Count',
            'Outcome',
        ])

        for GT_ID in range(NUM_CLASSES):
            for Pred_009_ID in range(NUM_CLASSES):
                for Pred_011_boundary_ID in range(NUM_CLASSES):
                    Count = int(
                        Transition_Counts[
                            GT_ID,
                            Pred_009_ID,
                            Pred_011_boundary_ID
                        ]
                    )

                    if Count == 0:
                        continue

                    Is_Correct_009 = GT_ID == Pred_009_ID
                    Is_Correct_011_boundary = GT_ID == Pred_011_boundary_ID

                    if not Is_Correct_009 and Is_Correct_011_boundary:
                        Outcome = 'Improved'
                    elif Is_Correct_009 and not Is_Correct_011_boundary:
                        Outcome = 'Regressed'
                    elif Is_Correct_009 and Is_Correct_011_boundary:
                        Outcome = 'Both_Correct'
                    elif Pred_009_ID == Pred_011_boundary_ID:
                        Outcome = 'Both_Wrong_Same_Prediction'
                    else:
                        Outcome = 'Both_Wrong_Different_Prediction'

                    Writer.writerow([
                        CLASS_NAMES[GT_ID],
                        CLASS_NAMES[Pred_009_ID],
                        CLASS_NAMES[Pred_011_boundary_ID],
                        Count,
                        Outcome,
                    ])

    # Save per-image summary
    with (Output_Dir / 'per_image_summary.csv').open(
        'w',
        newline='',
        encoding='utf-8-sig'
    ) as File:
        Writer = csv.writer(File)

        Writer.writerow([
            'Image',
            'Total_Pixels',
            'Improved_Pixels',
            'Regressed_Pixels',
            'Both_Correct_Pixels',
            'Both_Wrong_Pixels',
            'Same_Prediction_Pixels',
        ])

        Writer.writerows(Image_Summary_Rows)

    # Save class-level summary
    with (Output_Dir / 'per_class_transition_summary.csv').open(
        'w',
        newline='',
        encoding='utf-8-sig'
    ) as File:
        Writer = csv.writer(File)

        Writer.writerow([
            'Class',
            'Improved_Pixels',
            'Regressed_Pixels',
            'Net_Change',
            'Both_Correct_Pixels',
            'Both_Wrong_Pixels',
            'Same_Prediction_Pixels',
        ])

        for Class_ID, Class_Name in enumerate(CLASS_NAMES):
            Counts = Per_Class_Counts[Class_ID]

            Writer.writerow([
                Class_Name,
                Counts['Improved'],
                Counts['Regressed'],
                Counts['Improved'] - Counts['Regressed'],
                Counts['Both_Correct'],
                Counts['Both_Wrong'],
                Counts['Same_Prediction'],
            ])

    # Save class-level IoU comparison
    Class_IoU_Path = Output_Dir / 'class_iou_comparison.csv'

    with Class_IoU_Path.open(
        'w',
        newline='',
        encoding='utf-8-sig'
    ) as File:
        Writer = csv.writer(File)

        Writer.writerow([
            'Class',
            'EXP009_Mean_Per_Image_IoU',
            'EXP011_boundary_Mean_Per_Image_IoU',
            'Mean_Per_Image_IoU_Delta',
            'EXP009_Global_Pixel_IoU',
            'EXP011_boundary_Global_Pixel_IoU',
            'Global_Pixel_IoU_Delta',
            'EXP009_Image_Count',
            'EXP011_boundary_Image_Count',
        ])

        for Class_ID, Class_Name in enumerate(CLASS_NAMES):
            Mean_IoU_009 = (
                Per_Image_IoU_Sum_009[Class_ID]
                / Per_Image_IoU_Count_009[Class_ID]
                if Per_Image_IoU_Count_009[Class_ID] > 0
                else float('nan')
            )

            Mean_IoU_011_boundary = (
                Per_Image_IoU_Sum_011_boundary[Class_ID]
                / Per_Image_IoU_Count_011_boundary[Class_ID]
                if Per_Image_IoU_Count_011_boundary[Class_ID] > 0
                else float('nan')
            )

            Intersection_009 = Confusion_009[Class_ID, Class_ID]
            Union_009 = (
                Confusion_009[Class_ID, :].sum()
                + Confusion_009[:, Class_ID].sum()
                - Intersection_009
            )

            Intersection_011_boundary = Confusion_011_boundary[Class_ID, Class_ID]
            Union_011_boundary = (
                Confusion_011_boundary[Class_ID, :].sum()
                + Confusion_011_boundary[:, Class_ID].sum()
                - Intersection_011_boundary
            )

            Global_IoU_009 = (
                Intersection_009 / Union_009
                if Union_009 > 0
                else float('nan')
            )

            Global_IoU_011_boundary = (
                Intersection_011_boundary / Union_011_boundary
                if Union_011_boundary > 0
                else float('nan')
            )

            Writer.writerow([
                Class_Name,
                Mean_IoU_009,
                Mean_IoU_011_boundary,
                Mean_IoU_011_boundary - Mean_IoU_009,
                Global_IoU_009,
                Global_IoU_011_boundary,
                Global_IoU_011_boundary - Global_IoU_009,
                Per_Image_IoU_Count_009[Class_ID],
                Per_Image_IoU_Count_011_boundary[Class_ID],
            ])

    print(f'Class IoU comparison saved: {Class_IoU_Path}')

    # Console summary
    print('\n' + '=' * 75)
    print('FULL VALIDATION CLASS TRANSITION ANALYSIS')
    print('=' * 75)

    print(
        f'Validation Images: {len(Image_Paths)}'
    )

    print(
        f'Total Pixels: {int(Transition_Counts.sum()):,}'
    )

    print('\nPer-Class Summary')

    print(
        f'{"Class":<16}'
        f'{"Improved":>12}'
        f'{"Regressed":>12}'
        f'{"Net Change":>12}'
    )

    for Class_ID, Class_Name in enumerate(CLASS_NAMES):
        Counts = Per_Class_Counts[Class_ID]

        Net_Change = (
            Counts['Improved'] - Counts['Regressed']
        )

        print(
            f'{Class_Name:<16}'
            f'{Counts["Improved"]:>12,}'
            f'{Counts["Regressed"]:>12,}'
            f'{Net_Change:>12,}'
        )

    print('\nSaved transition analysis:')
    print(Combined_Path)
    print(Output_Dir / 'per_image_summary.csv')
    print(Output_Dir / 'per_class_transition_summary.csv')


# 6. Main
def Main():
    Parser = argparse.ArgumentParser()

    Parser.add_argument('--images', nargs='+', default=['A11_step2_sample4', 'A72_step4_sample4', 'A67_step1_sample1'], help='Validation image filename keywords.')

    Parser.add_argument('--limit', type=int, default=3, help='Maximum number of images to compare.')

    Args = Parser.parse_args()

    if Args.limit < 1:
        raise ValueError('--limit must be at least 1.')

    Device = torch.device(
        DEVICE
        if DEVICE != 'cuda' or torch.cuda.is_available()
        else 'cpu'
    )

    Checkpoint_Paths = {
        'exp009': (PROJECT_ROOT / 'outputs' / 'checkpoints' / 'exp009' / 'best_model.pth'),
        'exp011_boundary': (PROJECT_ROOT / 'outputs' / 'checkpoints' / 'exp011_boundary' / 'best_model.pth'),
    }

    for Experiment, Checkpoint_Path in Checkpoint_Paths.items():
        if not Checkpoint_Path.is_file():
            raise FileNotFoundError(f'{Experiment} checkpoint not found: {Checkpoint_Path}')

    # Load Validation Dataset
    Valid_Dataset, Valid_Dataloader = Get_Validation_Dataloader()

    Output_Dir = (PROJECT_ROOT / 'outputs' / 'visualizations' / 'exp009_vs_exp011_boundary' / 'pixel_comparison')

    Models = {}
    Predictions_By_Experiment = {}

    try:
        # Load Both Models
        for Experiment, Checkpoint_Path in Checkpoint_Paths.items():
            Checkpoint = torch.load(Checkpoint_Path, map_location=Device, weights_only=False)

            if isinstance(Checkpoint, dict) and 'model_state_dict' in Checkpoint:
                State_Dict = Checkpoint['model_state_dict']
            else:
                State_Dict = Checkpoint

            if Experiment == 'exp011_boundary':
                Base_Model = Get_Boundary_Model().to(Device)
            else:
                Base_Model = Get_Model().to(Device)

            Base_Model.load_state_dict(State_Dict)
            Base_Model.eval()

            if Experiment == 'exp011_boundary':
                Model = SegmentationOnlyModel(Base_Model).to(Device)
            else:
                Model = Base_Model

            Model.eval()
            Models[Experiment] = Model

            print('=' * 70)
            print(f'Loaded: {Experiment}')
            print(f'Checkpoint: {Checkpoint_Path}')

            if isinstance(Checkpoint, dict) and 'epoch' in Checkpoint:
                print(f'Checkpoint Epoch: {Checkpoint["epoch"]}')

        # Reconstruct full-resolution predictions
        for Experiment, Model in Models.items():
            print(f'\nReconstructing predictions: {Experiment}')

            Predictions_By_Experiment[Experiment] = (
                Reconstruct_Original_Predictions(
                    Model=Model,
                    Valid_Dataset=Valid_Dataset,
                    Valid_Dataloader=Valid_Dataloader,
                    Device=Device
                )
            )

        Predictions_009 = Predictions_By_Experiment['exp009']
        Predictions_011_boundary = Predictions_By_Experiment['exp011_boundary']

        # Full Validation Transition Analysis
        Transition_Output_Dir = (PROJECT_ROOT / 'outputs' / 'logs' / 'exp009_vs_exp011_boundary_transition_analysis'
)

        Analyze_Class_Transitions(
            Valid_Dataset=Valid_Dataset,
            Predictions_009=Predictions_009,
            Predictions_011_boundary=Predictions_011_boundary,
            Output_Dir=Transition_Output_Dir
        )

        # Match predictions by normalized absolute image path
        def Normalize_Path(Image_Path):
            return str(Path(Image_Path).resolve()).casefold()

        Prediction_009_By_Path = {
            Normalize_Path(Image_Path): Prediction
            for Image_Path, Prediction in Predictions_009.items()
        }

        Prediction_011_boundary_By_Path = {
            Normalize_Path(Image_Path): Prediction
            for Image_Path, Prediction in Predictions_011_boundary.items()
        }

        if set(Prediction_009_By_Path) != set(Prediction_011_boundary_By_Path):
            raise RuntimeError('EXP-009 and EXP-011_boundary prediction image sets do not match.')

        Saved_Count = 0

        for Image_Path in Valid_Dataset.Base_Dataset.Image_Paths:
            Image_Path = Path(Image_Path)
            Image_Key = Normalize_Path(Image_Path)

            if not any(
                Keyword.casefold() in Image_Path.stem.casefold()
                for Keyword in Args.images
            ):
                continue

            if Saved_Count >= Args.limit:
                break

            Target_Mask = Valid_Dataset.Get_Original_Mask(
                str(Image_Path)
            )

            Prediction_009 = Prediction_009_By_Path[Image_Key]
            Prediction_011_boundary = Prediction_011_boundary_By_Path[Image_Key]

            Output_Path = Save_Comparison_Visualization(
                Image_Path=Image_Path,
                Target_Mask=Target_Mask,
                Prediction_009=Prediction_009,
                Prediction_011_boundary=Prediction_011_boundary,
                Output_Dir=Output_Dir
            )

            print(f'Saved: {Output_Path}')
            Saved_Count += 1

        if Saved_Count == 0:
            raise RuntimeError(
                'No matching validation images found. '
                'Check --images against the validation filenames.'
            )

        print('\nComparison completed.')
        print(f'Images saved: {Saved_Count}')
        print(f'Output directory: {Output_Dir}')

    finally:
        Models.clear()
        Predictions_By_Experiment.clear()

        if Device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    Main()