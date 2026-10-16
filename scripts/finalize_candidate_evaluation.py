import csv
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEVICE
from src.model import Get_Model
from src.boundary_model import Get_Boundary_Model
from src.evaluate import Evaluate_Model


# ============================================================
# 1. Configuration
# ============================================================

CLASS_NAMES = [
    'Al',
    'Al3Ni',
    'Primary_Si',
    'Eutectic_Si',
]

CHECKPOINT_PATHS = {
    'exp007': PROJECT_ROOT / 'outputs' / 'checkpoints' / 'exp007' / 'best_model.pth',
    'exp008': PROJECT_ROOT / 'outputs' / 'checkpoints' / 'exp008' / 'best_model.pth',
    'exp009': PROJECT_ROOT / 'outputs' / 'checkpoints' / 'exp009' / 'best_model.pth',
    'exp010': PROJECT_ROOT / 'outputs' / 'checkpoints' / 'exp010' / 'best_model.pth',
    'exp011_boundary': PROJECT_ROOT / 'outputs' / 'checkpoints' / 'exp011_boundary' / 'best_model.pth',
    'exp012_boundary_weight_01': PROJECT_ROOT / 'outputs' / 'checkpoints' / 'exp012_boundary_weight_01' / 'best_model.pth',
}

BOUNDARY_EXPERIMENTS = {
    'exp011_boundary',
    'exp012_boundary_weight_01',
}

OUTPUT_DIR = (
    PROJECT_ROOT
    / 'outputs'
    / 'logs'
    / 'final_candidate_evaluation'
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

Device = torch.device(
    'cuda' if DEVICE == 'cuda' and torch.cuda.is_available() else 'cpu'
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
# 3. Utility Functions
# ============================================================

def To_Float(Value):
    if isinstance(Value, torch.Tensor):
        return float(Value.detach().cpu().item())

    return float(Value)


def To_Numpy(Value):
    if isinstance(Value, torch.Tensor):
        return Value.detach().cpu().numpy()

    return np.asarray(Value)


def Load_Checkpoint_Model(Experiment, Checkpoint_Path):
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

    if Experiment in BOUNDARY_EXPERIMENTS:
        Base_Model = Get_Boundary_Model().to(Device)
    else:
        Base_Model = Get_Model().to(Device)

    Base_Model.load_state_dict(State_Dict)
    Base_Model.eval()

    if Experiment in BOUNDARY_EXPERIMENTS:
        Model = SegmentationOnlyModel(Base_Model).to(Device)
    else:
        Model = Base_Model

    Model.eval()

    Stored_mIoU = None
    Epoch = None

    if isinstance(Checkpoint, dict):
        if 'validation_mIoU' in Checkpoint:
            Stored_mIoU = To_Float(
                Checkpoint['validation_mIoU']
            )

        if 'epoch' in Checkpoint:
            Epoch = int(Checkpoint['epoch'])

    print(f'Loaded: {Experiment}')
    print(f'Checkpoint: {Checkpoint_Path}')
    print(f'Checkpoint epoch: {Epoch}')
    print(f'Stored mIoU: {Stored_mIoU}')

    return Model, Stored_mIoU, Epoch


def Extract_Image_Metrics(Evaluation):
    Image_Metrics = Evaluation['Image_Metrics']
    Image_Results = {}

    for Image_Path, Metrics in Image_Metrics.items():
        Image_Name = Path(Image_Path).name

        Image_Results[Image_Name] = {
            'mIoU': To_Float(Metrics['mIoU']),
            'IoU': To_Numpy(Metrics['IoU']).astype(np.float64),
        }

    return Image_Results


def Calculate_Mean_Class_IoU(Image_Results):
    Class_Results = {}

    for Class_ID, Class_Name in enumerate(CLASS_NAMES):
        Values = [
            Metrics['IoU'][Class_ID]
            for Metrics in Image_Results.values()
        ]

        Values = np.asarray(Values, dtype=np.float64)
        Valid_Values = Values[np.isfinite(Values)]

        if len(Valid_Values) == 0:
            Class_Results[Class_Name] = float('nan')
        else:
            Class_Results[Class_Name] = float(
                Valid_Values.mean()
            )

    return Class_Results


def Write_CSV(File_Path, Field_Names, Rows):
    with File_Path.open(
        'w',
        newline='',
        encoding='utf-8-sig',
    ) as File:
        Writer = csv.DictWriter(
            File,
            fieldnames=Field_Names,
        )
        Writer.writeheader()
        Writer.writerows(Rows)


# ============================================================
# 4. Evaluation Pipeline
# ============================================================

def Main():
    print('=' * 80)
    print('Final Candidate Model Evaluation')
    print(f'Device: {Device}')
    print('Evaluation data: Validation only')
    print('=' * 80)

    Summary_Rows = []
    Per_Image_Rows = []
    Per_Class_Rows = []

    All_Image_Results = {}
    Recomputed_Scores = {}

    for Index, (Experiment, Checkpoint_Path) in enumerate(
        CHECKPOINT_PATHS.items(),
        start=1,
    ):
        print('\n' + '=' * 80)
        print(
            f'[{Index}/{len(CHECKPOINT_PATHS)}] '
            f'Evaluating {Experiment}'
        )
        print('=' * 80)

        Model, Stored_mIoU, Epoch = Load_Checkpoint_Model(
            Experiment=Experiment,
            Checkpoint_Path=Checkpoint_Path,
        )

        with torch.inference_mode():
            Evaluation = Evaluate_Model(
                Model=Model,
                Device=Device,
            )

        Recomputed_mIoU = To_Float(
            Evaluation['Validation_mIoU']
        )

        Image_Results = Extract_Image_Metrics(Evaluation)
        Class_Results = Calculate_Mean_Class_IoU(
            Image_Results
        )

        All_Image_Results[Experiment] = Image_Results
        Recomputed_Scores[Experiment] = Recomputed_mIoU

        Summary_Rows.append({
            'experiment': Experiment,
            'checkpoint': str(Checkpoint_Path.relative_to(PROJECT_ROOT)),
            'epoch': Epoch,
            'stored_validation_mIoU': Stored_mIoU,
            'recomputed_validation_mIoU': Recomputed_mIoU,
            'difference_from_stored': (
                abs(Stored_mIoU - Recomputed_mIoU)
                if Stored_mIoU is not None
                else None
            ),
            'validation_image_count': len(Image_Results),
        })

        for Class_Name, Class_IoU in Class_Results.items():
            Per_Class_Rows.append({
                'experiment': Experiment,
                'class_name': Class_Name,
                'mean_class_IoU': Class_IoU,
            })

        for Image_Name, Metrics in Image_Results.items():
            Row = {
                'experiment': Experiment,
                'image_name': Image_Name,
                'image_mIoU': Metrics['mIoU'],
            }

            for Class_ID, Class_Name in enumerate(CLASS_NAMES):
                Row[f'{Class_Name}_IoU'] = Metrics['IoU'][Class_ID]

            Per_Image_Rows.append(Row)

        print(f'Recomputed Validation mIoU: {Recomputed_mIoU:.6f}')

        del Model
        del Evaluation

        if Device.type == 'cuda':
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # 5. Validate Image Set Consistency
    # --------------------------------------------------------

    Reference_Experiment = 'exp009'
    Reference_Images = set(
        All_Image_Results[Reference_Experiment].keys()
    )

    for Experiment, Image_Results in All_Image_Results.items():
        Current_Images = set(Image_Results.keys())

        if Current_Images != Reference_Images:
            Missing_Images = sorted(Reference_Images - Current_Images)
            Extra_Images = sorted(Current_Images - Reference_Images)

            raise RuntimeError(
                f'Validation image mismatch: {Experiment}\n'
                f'Missing: {Missing_Images}\n'
                f'Extra: {Extra_Images}'
            )

    # --------------------------------------------------------
    # 6. Write Evaluation Results
    # --------------------------------------------------------

    Write_CSV(
        OUTPUT_DIR / 'candidate_summary.csv',
        [
            'experiment',
            'checkpoint',
            'epoch',
            'stored_validation_mIoU',
            'recomputed_validation_mIoU',
            'difference_from_stored',
            'validation_image_count',
        ],
        Summary_Rows,
    )

    Write_CSV(
        OUTPUT_DIR / 'per_class_iou.csv',
        [
            'experiment',
            'class_name',
            'mean_class_IoU',
        ],
        Per_Class_Rows,
    )

    Per_Image_Fields = [
        'experiment',
        'image_name',
        'image_mIoU',
    ]

    Per_Image_Fields.extend(
        [f'{Class_Name}_IoU' for Class_Name in CLASS_NAMES]
    )

    Write_CSV(
        OUTPUT_DIR / 'per_image_metrics.csv',
        Per_Image_Fields,
        Per_Image_Rows,
    )

    # --------------------------------------------------------
    # 7. Print Final Comparison
    # --------------------------------------------------------

    Sorted_Experiments = sorted(
        Recomputed_Scores,
        key=Recomputed_Scores.get,
        reverse=True,
    )

    print('\n' + '=' * 90)
    print('FINAL VALIDATION mIoU RANKING')
    print('=' * 90)
    print(f'{"Rank":<8}{"Experiment":<34}{"Validation mIoU":>20}')
    print('-' * 65)

    for Rank, Experiment in enumerate(
        Sorted_Experiments,
        start=1,
    ):
        print(
            f'{Rank:<8}'
            f'{Experiment:<34}'
            f'{Recomputed_Scores[Experiment]:>20.6f}'
        )

    print('\n' + '=' * 90)
    print('CLASS IoU SUMMARY')
    print('=' * 90)

    Class_Summaries = {
        Experiment: Calculate_Mean_Class_IoU(Image_Results)
        for Experiment, Image_Results in All_Image_Results.items()
    }

    print(f'{"Experiment":<34}', end='')

    for Class_Name in CLASS_NAMES:
        print(f'{Class_Name:>16}', end='')

    print()
    print('-' * 98)

    for Experiment in CHECKPOINT_PATHS:
        print(f'{Experiment:<34}', end='')

        for Class_Name in CLASS_NAMES:
            Value = Class_Summaries[Experiment][Class_Name]

            if np.isfinite(Value):
                print(f'{Value:>16.6f}', end='')
            else:
                print(f'{"N/A":>16}', end='')

        print()

    # Per-image score changes relative to EXP-009.
    print('\n' + '=' * 90)
    print('PER-IMAGE mIoU CHANGE RELATIVE TO EXP-009')
    print('=' * 90)

    for Experiment in CHECKPOINT_PATHS:
        if Experiment == Reference_Experiment:
            continue

        Changes = []

        for Image_Name in sorted(Reference_Images):
            Baseline_Value = All_Image_Results[
                Reference_Experiment
            ][Image_Name]['mIoU']

            Current_Value = All_Image_Results[
                Experiment
            ][Image_Name]['mIoU']

            Changes.append(Current_Value - Baseline_Value)

        Improved_Count = sum(Value > 0 for Value in Changes)
        Regressed_Count = sum(Value < 0 for Value in Changes)
        Equal_Count = sum(Value == 0 for Value in Changes)

        print(
            f'{Experiment}: '
            f'mIoU delta={Recomputed_Scores[Experiment] - Recomputed_Scores[Reference_Experiment]:+.6f}, '
            f'improved images={Improved_Count}, '
            f'regressed images={Regressed_Count}, '
            f'equal images={Equal_Count}'
        )

    print('\nSaved results:')
    print(OUTPUT_DIR / 'candidate_summary.csv')
    print(OUTPUT_DIR / 'per_class_iou.csv')
    print(OUTPUT_DIR / 'per_image_metrics.csv')
    print('\nNo training was performed.')
    print('Test images were not used for model selection.')


if __name__ == '__main__':
    Main()