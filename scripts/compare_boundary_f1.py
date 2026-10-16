# scripts/compare_boundary_f1.py
# EXP-009 vs EXP-011 Boundary F1 Comparison

from pathlib import Path
import csv
import sys

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEVICE, NUM_CLASSES
from src.model import Get_Model
from src.boundary_model import Get_Boundary_Model
from src.evaluate import (
    Get_Validation_Dataloader,
    Reconstruct_Original_Predictions,
)


# ============================================================
# 1. Experiment Configuration
# ============================================================

Checkpoint_Paths = {
    'exp009': (
        PROJECT_ROOT
        / 'outputs'
        / 'checkpoints'
        / 'exp009'
        / 'best_model.pth'
    ),
    'exp011_boundary': (
        PROJECT_ROOT
        / 'outputs'
        / 'checkpoints'
        / 'exp011_boundary'
        / 'best_model.pth'
    ),
}

Tolerance_Pixels = [0, 1, 2]

Output_Dir = (
    PROJECT_ROOT
    / 'outputs'
    / 'logs'
    / 'exp009_vs_exp011_boundary_f1'
)


# ============================================================
# 2. Segmentation-Only Wrapper
# ============================================================

class SegmentationOnlyModel(torch.nn.Module):
    def __init__(self, Model):
        super().__init__()
        self.Model = Model

    def forward(self, Images):
        Outputs = self.Model(Images)
        return Outputs['segmentation']


# ============================================================
# 3. Load Checkpoint
# ============================================================

def Load_Checkpoint(Experiment, Checkpoint_Path, Device):
    if not Checkpoint_Path.is_file():
        raise FileNotFoundError(
            f'Checkpoint not found: {Checkpoint_Path}'
        )

    Checkpoint = torch.load(
        Checkpoint_Path,
        map_location=Device,
        weights_only=False,
    )

    if (
        isinstance(Checkpoint, dict)
        and 'model_state_dict' in Checkpoint
    ):
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

    print(f'Loaded: {Experiment}')
    print(f'Checkpoint: {Checkpoint_Path}')

    return Model


# ============================================================
# 4. Extract Class-Agnostic Boundaries
# ============================================================

def Extract_Boundary(Mask):
    """
    Mask: [H, W] class-ID mask.

    A pixel is considered a boundary if its class differs
    from at least one of its horizontal or vertical neighbors.
    Both sides of each class transition are marked.
    """

    Mask = np.asarray(Mask)

    if Mask.ndim != 2:
        raise ValueError(
            f'Expected 2D mask, got {Mask.shape}'
        )

    Boundary = np.zeros(Mask.shape, dtype=bool)

    # Horizontal transitions
    Horizontal_Change = Mask[:, 1:] != Mask[:, :-1]

    Boundary[:, 1:] |= Horizontal_Change
    Boundary[:, :-1] |= Horizontal_Change

    # Vertical transitions
    Vertical_Change = Mask[1:, :] != Mask[:-1, :]

    Boundary[1:, :] |= Vertical_Change
    Boundary[:-1, :] |= Vertical_Change

    return Boundary


# ============================================================
# 5. Boundary Matching With Tolerance
# ============================================================

def Dilate_Boundary(Boundary, Tolerance):
    """
    Tolerance=0: exact boundary pixel match.
    Tolerance=1 or 2: match within the corresponding
    square neighborhood using max pooling.
    """

    Boundary_Tensor = torch.from_numpy(
        Boundary.astype(np.float32)
    )[None, None, :, :]

    if Tolerance == 0:
        return Boundary

    Kernel_Size = 2 * Tolerance + 1

    Dilated = F.max_pool2d(
        Boundary_Tensor,
        kernel_size=Kernel_Size,
        stride=1,
        padding=Tolerance,
    )

    return Dilated[0, 0].numpy() > 0


def Calculate_Boundary_Metrics(
    Target_Mask,
    Prediction_Mask,
    Tolerance,
):
    Target_Boundary = Extract_Boundary(Target_Mask)
    Prediction_Boundary = Extract_Boundary(Prediction_Mask)

    Target_Count = int(Target_Boundary.sum())
    Prediction_Count = int(Prediction_Boundary.sum())

    # Empty-boundary cases are reported as undefined.
    if Target_Count == 0 or Prediction_Count == 0:
        return {
            'Precision': float('nan'),
            'Recall': float('nan'),
            'F1': float('nan'),
            'Target_Boundary_Pixels': Target_Count,
            'Prediction_Boundary_Pixels': Prediction_Count,
        }

    Target_Dilated = Dilate_Boundary(
        Target_Boundary,
        Tolerance,
    )

    Prediction_Dilated = Dilate_Boundary(
        Prediction_Boundary,
        Tolerance,
    )

    # Predicted pixels matched to a target boundary.
    Matched_Prediction = int(
        (Prediction_Boundary & Target_Dilated).sum()
    )

    # Target pixels matched to a predicted boundary.
    Matched_Target = int(
        (Target_Boundary & Prediction_Dilated).sum()
    )

    Precision = Matched_Prediction / Prediction_Count
    Recall = Matched_Target / Target_Count

    F1 = (
        2 * Precision * Recall / (Precision + Recall)
        if Precision + Recall > 0
        else 0.0
    )

    return {
        'Precision': Precision,
        'Recall': Recall,
        'F1': F1,
        'Target_Boundary_Pixels': Target_Count,
        'Prediction_Boundary_Pixels': Prediction_Count,
    }


# ============================================================
# 6. Main
# ============================================================

def Main():
    Device = torch.device(
        DEVICE
        if DEVICE != 'cuda' or torch.cuda.is_available()
        else 'cpu'
    )

    print('=' * 72)
    print('EXP-009 vs EXP-011 Boundary F1 Comparison')
    print(f'Device: {Device}')
    print(f'Tolerances: {Tolerance_Pixels}')
    print('=' * 72)

    # Load the same original-image validation set for both models.
    Valid_Dataset, Valid_Dataloader = (
        Get_Validation_Dataloader()
    )

    Models = {}
    Predictions_By_Experiment = {}

    try:
        for Experiment, Checkpoint_Path in Checkpoint_Paths.items():
            Models[Experiment] = Load_Checkpoint(
                Experiment,
                Checkpoint_Path,
                Device,
            )

        # Reconstruct full-resolution segmentation predictions.
        for Experiment, Model in Models.items():
            print(f'\nReconstructing: {Experiment}')

            Predictions_By_Experiment[Experiment] = (
                Reconstruct_Original_Predictions(
                    Model=Model,
                    Valid_Dataset=Valid_Dataset,
                    Valid_Dataloader=Valid_Dataloader,
                    Device=Device,
                )
            )

        def Normalize_Path(Image_Path):
            return str(Path(Image_Path).resolve()).casefold()

        Prediction_Maps = {}

        for Experiment, Predictions in (
            Predictions_By_Experiment.items()
        ):
            Prediction_Maps[Experiment] = {
                Normalize_Path(Image_Path): (
                    Prediction.detach().cpu().numpy()
                    if isinstance(Prediction, torch.Tensor)
                    else np.asarray(Prediction)
                )
                for Image_Path, Prediction in Predictions.items()
            }

        if (
            set(Prediction_Maps['exp009'])
            != set(Prediction_Maps['exp011_boundary'])
        ):
            raise RuntimeError(
                'The two models have different validation image sets.'
            )

        Output_Dir.mkdir(parents=True, exist_ok=True)

        Image_Rows = []
        Summary_Rows = []

        for Tolerance in Tolerance_Pixels:
            print('\n' + '=' * 72)
            print(f'Boundary tolerance: {Tolerance} pixel(s)')
            print('=' * 72)

            Experiment_Scores = {
                Experiment: {
                    'Precision': [],
                    'Recall': [],
                    'F1': [],
                }
                for Experiment in Checkpoint_Paths
            }

            for Image_Path in Valid_Dataset.Base_Dataset.Image_Paths:
                Image_Path = Path(Image_Path)
                Image_Key = Normalize_Path(Image_Path)

                Target_Mask = Valid_Dataset.Get_Original_Mask(
                    str(Image_Path)
                )

                Target = (
                    Target_Mask.detach().cpu().numpy()
                    if isinstance(Target_Mask, torch.Tensor)
                    else np.asarray(Target_Mask)
                ).astype(np.int64)

                if (
                    Target.ndim != 2
                    or Target.min() < 0
                    or Target.max() >= NUM_CLASSES
                ):
                    raise ValueError(
                        f'Invalid target mask: {Image_Path.name}'
                    )

                for Experiment in Checkpoint_Paths:
                    Prediction = Prediction_Maps[
                        Experiment
                    ][Image_Key].astype(np.int64)

                    if Prediction.shape != Target.shape:
                        raise ValueError(
                            f'Shape mismatch: {Image_Path.name}'
                        )

                    Metrics = Calculate_Boundary_Metrics(
                        Target,
                        Prediction,
                        Tolerance,
                    )

                    for Metric_Name in (
                        'Precision',
                        'Recall',
                        'F1',
                    ):
                        Value = Metrics[Metric_Name]

                        if np.isfinite(Value):
                            Experiment_Scores[
                                Experiment
                            ][Metric_Name].append(Value)

                    Image_Rows.append([
                        Tolerance,
                        Image_Path.name,
                        Experiment,
                        Metrics['Precision'],
                        Metrics['Recall'],
                        Metrics['F1'],
                        Metrics['Target_Boundary_Pixels'],
                        Metrics['Prediction_Boundary_Pixels'],
                    ])

            # Macro average across images with defined metrics.
            for Experiment in Checkpoint_Paths:
                Scores = Experiment_Scores[Experiment]

                Mean_Precision = (
                    float(np.mean(Scores['Precision']))
                    if Scores['Precision'] else float('nan')
                )

                Mean_Recall = (
                    float(np.mean(Scores['Recall']))
                    if Scores['Recall'] else float('nan')
                )

                Mean_F1 = (
                    float(np.mean(Scores['F1']))
                    if Scores['F1'] else float('nan')
                )

                Summary_Rows.append([
                    Tolerance,
                    Experiment,
                    Mean_Precision,
                    Mean_Recall,
                    Mean_F1,
                    len(Scores['F1']),
                ])

                print(
                    f'{Experiment:<20} '
                    f'Precision={Mean_Precision:.6f} '
                    f'Recall={Mean_Recall:.6f} '
                    f'F1={Mean_F1:.6f} '
                    f'Images={len(Scores["F1"])}'
                )

            Score_Map = {
                Row[1]: Row[4]
                for Row in Summary_Rows
                if Row[0] == Tolerance
            }

            Score_009 = Score_Map['exp009']
            Score_011 = Score_Map['exp011_boundary']

            print(
                'F1 Delta (EXP-011 - EXP-009): '
                f'{Score_011 - Score_009:+.6f}'
            )

        # Save per-image metrics.
        with (
            Output_Dir / 'per_image_boundary_metrics.csv'
        ).open(
            'w',
            newline='',
            encoding='utf-8-sig',
        ) as File:
            Writer = csv.writer(File)

            Writer.writerow([
                'Tolerance_Pixels',
                'Image',
                'Experiment',
                'Boundary_Precision',
                'Boundary_Recall',
                'Boundary_F1',
                'Target_Boundary_Pixels',
                'Prediction_Boundary_Pixels',
            ])

            Writer.writerows(Image_Rows)

        # Save macro-average metrics.
        with (
            Output_Dir / 'boundary_summary.csv'
        ).open(
            'w',
            newline='',
            encoding='utf-8-sig',
        ) as File:
            Writer = csv.writer(File)

            Writer.writerow([
                'Tolerance_Pixels',
                'Experiment',
                'Mean_Boundary_Precision',
                'Mean_Boundary_Recall',
                'Mean_Boundary_F1',
                'Images_With_Defined_F1',
            ])

            Writer.writerows(Summary_Rows)

        print('\n' + '=' * 72)
        print('Boundary evaluation completed.')
        print(f'Per-image results: {Output_Dir / "per_image_boundary_metrics.csv"}')
        print(f'Summary: {Output_Dir / "boundary_summary.csv"}')
        print('Checkpoint files were only read, not overwritten.')

    finally:
        Models.clear()
        Predictions_By_Experiment.clear()

        if Device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    Main()
