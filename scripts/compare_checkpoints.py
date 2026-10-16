import argparse
import sys

from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEVICE
from src.model import Get_Model
from src.evaluate import Evaluate_Model


# 1. Class configuration
CLASS_NAMES = [
    'Al',
    'Al3Ni',
    'Primary_Si',
    'Eutectic_Si'
]


#2. Load Trained Model
def Load_Checkpoint_Model(Checkpoint_Path, Device):
    Checkpoint_Path = Path(Checkpoint_Path)

    if not Checkpoint_Path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {Checkpoint_Path}')

    Model = Get_Model().to(Device)

    Checkpoint = torch.load(Checkpoint_Path, map_location=Device, weights_only=False)

    if not isinstance(Checkpoint, dict) or 'model_state_dict' not in Checkpoint:
        raise KeyError(f'model_state_dict not fount: {Checkpoint_Path}')

    Model.load_state_dict(Checkpoint['model_state_dict'])

    Model.eval()

    print(f'Loaded Checkpoint: {Checkpoint_Path}')

    if 'epoch' in Checkpoint:
        print(f'Checkpoint Epoch: {Checkpoint["epoch"]}')

    if 'validation_mIoU' in Checkpoint:
        print(f'Checkpoint Validation mIoU: {Checkpoint["validation_mIoU"]:.6f}')

    return Model


# 3. Extract Evaluation Results
def Extract_Results(Results):
    Image_Metrics = Results['Image_Metrics']

    Image_Result_Dict = {}

    for Image_Path, Metrics in Image_Metrics.items():
        Image_Name = Path(Image_Path).name

        Image_Result_Dict[Image_Name] = {
            'IoU': Metrics['IoU'].detach().cpu().numpy(),
            'mIoU': float(Metrics['mIoU'].item())
        }

    return Image_Result_Dict


# 4. Calculate Mean Class IoU
def Calculate_Mean_Class_IoU(Image_Result_Dict):
    Class_IoU_Values = []

    for Class_ID in range(len(CLASS_NAMES)):
        IoU_List = [
            Metrics['IoU'][Class_ID]
            for Metrics in Image_Result_Dict.values()
        ]

        IoU_Array = np.asarray(IoU_List, dtype=np.float64)

        Valid_IoU = IoU_Array[np.isfinite(IoU_Array)]

        if len(Valid_IoU) == 0:
            Mean_IoU = float('nan')
        else:
            Mean_IoU = float(Valid_IoU.mean())

        Class_IoU_Values.append(Mean_IoU)
    
    return Class_IoU_Values


# 5. Print Comparison Results
def Compare_Results(Model_Results, Model_mIoU):
    Experiment_Names = list(Model_Results.keys())

    if len(Experiment_Names) < 2:
        raise ValueError('At least two models are required for comparison')

    Baseline_Name = 'exp004'

    if Baseline_Name not in Model_Results:
        raise KeyError(f'Baseline model not found: { Baseline_Name}')
    
    Baseline_Image_Names = set(Model_Results[Baseline_Name].keys())

    for Experiment in Experiment_Names:
        if set(Model_Results[Experiment].keys()) != Baseline_Image_Names:
            raise RuntimeError(f'{Experiment} was not evaluated on the same images as {Baseline_Name}')

    Class_IoU_Results = {
        Experiment: Calculate_Mean_Class_IoU(Model_Results[Experiment])
        for Experiment in Experiment_Names
    }

    # Class-Level IoU Comparison
    print('\n' + '=' * 100)
    print('Class IoU Comparison')
    print('=' * 100)

    Header = f'{"Class":<18}'

    for Experiment in Experiment_Names:
        Header += f'{Experiment:>14}'

    print(Header)
    print('-' * len(Header))

    for Class_ID, Class_Name in enumerate(CLASS_NAMES):
        Row = f'{Class_Name:<18}'

        for Experiment in Experiment_Names:
            IoU_Value = Class_IoU_Results[Experiment][Class_ID]

            IoU_Text = (
                f'{IoU_Value:.6f}'
                if np.isfinite(IoU_Value)
                else 'N/A'
            )

            Row += f'{IoU_Text:>14}'
        
        print(Row)
        
    # Overall Validatioin mIoU
    print('\n' + '=' * 100)
    print('Overall Validation mIoU Comparison')
    print('=' * 100)
        
    Baseline_mIoU = Model_mIoU[Baseline_Name]

    print(f'{"Experiment":<18}{"Validation mIoU":>20}{"Change vs EXP-004":>22}')
    print('-' * 60)

    for Experiment in Experiment_Names:
        Current_mIoU = Model_mIoU[Experiment]
        Difference = Current_mIoU - Baseline_mIoU

        print(
            f'{Experiment:<18}'
            f'{Current_mIoU:>20.6f}'
            f'{Difference:>+22.6f}'
        )

    # Image-Level Comparison
    print('\n' + '=' * 110)
    print('Image-Level mIoU Comparison')
    print('=' * 110)

    Header = f'{"Image":<38}'

    for Experiment in Experiment_Names:
        Header += f'{Experiment:>14}'

    print(Header)
    print('-' * len(Header))

    for Image_Name in sorted(Baseline_Image_Names):
        Row = f'{Image_Name:<38}'

        for Experiment in Experiment_Names:
            Image_mIoU = Model_Results[Experiment][Image_Name]['mIoU']
            Row += f'{Image_mIoU:>14.6f}'

        print(Row)

    print('=' * 110)
    


# 6. Main Comparison Pipeline
def Compare_Checkpoints(Checkpoint_Paths):
    if DEVICE == 'cuda' and torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    print(f'Using Device: {Device}')

    Model_Results = {}
    Model_mIoU = {}

    for Index, (Experiment, Checkpoint_Path) in enumerate(Checkpoint_Paths.items(), start=1):
        print('\n' + '=' * 70)
        print(f'[{Index}/{len(Checkpoint_Paths)}] Evaluating {Experiment}')
        print('=' * 70)

        Model = Load_Checkpoint_Model(
            Checkpoint_Path=Checkpoint_Path,
            Device=Device
        )

        Results_Raw = Evaluate_Model(
            Model=Model,
            Device=Device
        )

        Model_mIoU[Experiment] = float(Results_Raw['Validation_mIoU'].item())

        Model_Results[Experiment] = Extract_Results(Results_Raw)

        del Model
        del Results_Raw

        if Device.type == 'cuda':
            torch.cuda.empty_cache()

    Compare_Results(
        Model_Results=Model_Results,
        Model_mIoU=Model_mIoU
    )


# 7. Command Line Entry Point
if __name__ == '__main__':
    Parser = argparse.ArgumentParser(description='Compare KoMaP segmentation checkpoints')

    Parser.add_argument('--baseline', type=str, default='outputs/checkpoints/best_model.pth', help='EXP-004 baseline checkpoint path')

    Parser.add_argument('--weighted', type=str, default='outputs/checkpoints/exp005/best_model.pth', help='EXP-005 weighted-loss checkpoint path')

    Parser.add_argument('--finetuned', type=str, default='outputs/checkpoints/exp006/best_model.pth', help='EXP-006 fine_tuned checkpoint path')

    Args = Parser.parse_args()

    Checkpoint_Paths = {
        'exp004': Path(Args.baseline),
        'exp005': Path(Args.weighted),
        'exp006': Path(Args.finetuned)
    }

    for Experiment, Checkpoint_Path in Checkpoint_Paths.items():
        if not Checkpoint_Path.is_file():
            raise FileNotFoundError(f'{Experiment} checkpoint not found: {Checkpoint_Path}')


    Compare_Checkpoints(
        Checkpoint_Paths=Checkpoint_Paths
    )