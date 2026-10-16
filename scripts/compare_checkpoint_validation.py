from pathlib import Path
import csv
import json

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.config import DEVICE, NUM_CLASSES
from src.model import Get_Model
from src.boundary_model import Get_Boundary_Model
from src.evaluate import Evaluate_Model


# 1. Checkpoint Configuration
Checkpoint_Paths = {
    'exp009': (
        PROJECT_ROOT / 'outputs' / 'checkpoints'
        / 'exp009' / 'best_model.pth'
    ),
    'exp011_boundary': (
        PROJECT_ROOT / 'outputs' / 'checkpoints'
        / 'exp011_boundary' / 'best_model.pth'
    ),
}

Expected_Scores = {
    'exp009': 0.747717,
    'exp011_boundary': 0.748260,
}

Class_Names = {
    0: 'Al',
    1: 'Al3Ni',
    2: 'Primary_Si',
    3: 'Eutectic_Si',
}

Device = torch.device('cpu')

Output_Dir = PROJECT_ROOT / 'outputs' / 'analysis' / 'validation'
Output_Dir.mkdir(parents=True, exist_ok=True)


class SegmentationOnlyModel(torch.nn.Module):
    def __init__(self, Model):
        super().__init__()
        self.Model = Model

    def forward(self, Images):
        Outputs = self.Model(Images)
        return Outputs['segmentation']


def Load_Checkpoint(Experiment, Checkpoint_Path):
    Checkpoint = torch.load(
        Checkpoint_Path,
        map_location=Device,
        weights_only=False
    )

    if Experiment == 'exp011_boundary':
        Base_Model = Get_Boundary_Model().to(Device)
    else:
        Base_Model = Get_Model().to(Device)

    State_Dict = (
        Checkpoint['model_state_dict']
        if isinstance(Checkpoint, dict)
        and 'model_state_dict' in Checkpoint
        else Checkpoint
    )

    Base_Model.load_state_dict(State_Dict)
    Base_Model.eval()

    if Experiment == 'exp011_boundary':
        Model = SegmentationOnlyModel(Base_Model).to(Device)
    else:
        Model = Base_Model

    Model.eval()

    Stored_mIoU = (
        float(Checkpoint['validation_mIoU'])
        if isinstance(Checkpoint, dict)
        and 'validation_mIoU' in Checkpoint
        else None
    )

    return Model, Checkpoint, Stored_mIoU


def Analyze_Class_IoU(Image_Metrics):
    Class_IoU_Values = {
        Class_ID: []
        for Class_ID in range(NUM_CLASSES)
    }

    Image_Rows = []

    for Image_Path, Metrics in Image_Metrics.items():
        IoU_List = Metrics['IoU'].detach().cpu().tolist()
        Image_mIoU = float(Metrics['mIoU'].detach().cpu().item())

        Row = {
            'Image': Path(Image_Path).name,
            'mIoU': Image_mIoU,
        }

        for Class_ID in range(NUM_CLASSES):
            IoU_Value = float(IoU_List[Class_ID])

            Row[Class_Names[Class_ID] + '_IoU'] = (
                IoU_Value if IoU_Value == IoU_Value else None
            )

            # NaN IoU는 기존 metric에서 제외되므로 평균에서도 제외
            if IoU_Value == IoU_Value:
                Class_IoU_Values[Class_ID].append(IoU_Value)

        Image_Rows.append(Row)

    Class_Summary = {}

    for Class_ID, Values in Class_IoU_Values.items():
        Class_Summary[Class_Names[Class_ID]] = (
            sum(Values) / len(Values)
            if Values else None
        )

    return Class_Summary, Image_Rows


def Save_Image_CSV(Experiment, Image_Rows):
    CSV_Path = Output_Dir / f'{Experiment}_image_metrics.csv'

    Field_Names = [
        'Image',
        'mIoU',
        *[
            Class_Names[Class_ID] + '_IoU'
            for Class_ID in range(NUM_CLASSES)
        ],
    ]

    with CSV_Path.open(
        'w',
        newline='',
        encoding='utf-8-sig'
    ) as File:
        Writer = csv.DictWriter(File, fieldnames=Field_Names)
        Writer.writeheader()
        Writer.writerows(Image_Rows)

    return CSV_Path


def Main():
    Results = {}

    print('=' * 70)
    print('EXP-009 vs EXP-011_boundary Validation Analysis')
    print(f'Device: {Device}')
    print('=' * 70)

    for Experiment, Checkpoint_Path in Checkpoint_Paths.items():
        if not Checkpoint_Path.is_file():
            raise FileNotFoundError(
                f'{Experiment} checkpoint not found: {Checkpoint_Path}'
            )

        Model, Checkpoint, Stored_mIoU = Load_Checkpoint(
            Experiment,
            Checkpoint_Path
        )

        print('\n' + '-' * 70)
        print(f'Experiment: {Experiment}')
        print(f'Checkpoint: {Checkpoint_Path}')
        print(f'Checkpoint Epoch: {Checkpoint.get("epoch", "Unknown")}')

        # 기존 검증 파이프라인 사용
        Evaluation = Evaluate_Model(Model=Model, Device=Device)
        Recomputed_mIoU = float(
            Evaluation['Validation_mIoU'].detach().cpu().item()
        )

        Class_Summary, Image_Rows = Analyze_Class_IoU(
            Evaluation['Image_Metrics']
        )

        CSV_Path = Save_Image_CSV(Experiment, Image_Rows)

        Results[Experiment] = {
            'Stored_mIoU': Stored_mIoU,
            'Recomputed_mIoU': Recomputed_mIoU,
            'Expected_mIoU': Expected_Scores[Experiment],
            'Class_IoU_Mean_Per_Image': Class_Summary,
            'Image_Count': len(Image_Rows),
            'Image_CSV': str(CSV_Path),
        }

        print('\nClass IoU Mean Across Images')
        for Class_Name, IoU_Value in Class_Summary.items():
            if IoU_Value is None:
                print(f'{Class_Name}: N/A')
            else:
                print(f'{Class_Name}: {IoU_Value:.6f}')

        print(f'\nRecomputed Validation mIoU: {Recomputed_mIoU:.6f}')
        print(f'Expected Validation mIoU:  {Expected_Scores[Experiment]:.6f}')
        print(
            'Difference: '
            f'{abs(Recomputed_mIoU - Expected_Scores[Experiment]):.8f}'
        )
        print(f'Image metrics CSV: {CSV_Path}')

        del Model

        if Device.type == 'cuda':
            torch.cuda.empty_cache()

    Score_009 = Results['exp009']['Recomputed_mIoU']
    Score_011 = Results['exp011_boundary']['Recomputed_mIoU']

    print('\n' + '=' * 70)
    print('FINAL COMPARISON')
    print('=' * 70)
    print(f'EXP-009 mIoU: {Score_009:.6f}')
    print(f'EXP-011_boundary mIoU: {Score_011:.6f}')
    print(f'Delta (011 - 009): {Score_011 - Score_009:+.6f}')

    JSON_Path = Output_Dir / 'checkpoint_validation_analysis.json'

    with JSON_Path.open('w', encoding='utf-8') as File:
        json.dump(Results, File, ensure_ascii=False, indent=4, allow_nan=False)

    print(f'\nJSON summary saved: {JSON_Path}')
    print('Checkpoint files were only read, not overwritten.')


if __name__ == '__main__':
    Main()