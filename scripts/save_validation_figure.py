from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.config import DEVICE, NUM_CLASSES
from src.boundary_model import Get_Boundary_Model
from src.evaluate import (
    Evaluate_Model,
    Get_Validation_Dataloader,
)


# 1. Configuration
Checkpoint_Path = (
    PROJECT_ROOT
    / 'outputs'
    / 'checkpoints'
    / 'exp011_boundary'
    / 'best_model.pth'
)

Output_Dir = PROJECT_ROOT / 'outputs' / 'analysis' / 'figures'
Output_Dir.mkdir(parents=True, exist_ok=True)

Target_Image_Name = 'A117_step4_sample2_image.png'

# Class ID별 RGB 색상: 프로젝트의 기존 마스크 색상 체계
Class_Colors = np.array([
    [153, 127, 76],  # 0: Al
    [25, 76, 153],   # 1: Al3Ni
    [76, 178, 76],   # 2: Primary Si
    [204, 204, 204], # 3: Eutectic Si
], dtype=np.uint8)

Class_Names = ['Al', 'Al3Ni', 'Primary Si', 'Eutectic Si']

Device = torch.device(
    DEVICE if DEVICE != 'cuda' or torch.cuda.is_available()
    else 'cpu'
)


class SegmentationOnlyModel(torch.nn.Module):
    def __init__(self, Model):
        super().__init__()
        self.Model = Model

    def forward(self, Images):
        Outputs = self.Model(Images)
        return Outputs['segmentation']


def Main():
    if not Checkpoint_Path.is_file():
        raise FileNotFoundError(Checkpoint_Path)

    Checkpoint = torch.load(
        Checkpoint_Path,
        map_location=Device,
        weights_only=False
    )

    Base_Model = Get_Boundary_Model().to(Device)

    State_Dict = (
        Checkpoint['model_state_dict']
        if isinstance(Checkpoint, dict)
        and 'model_state_dict' in Checkpoint
        else Checkpoint
    )

    Base_Model.load_state_dict(State_Dict)

    Model = SegmentationOnlyModel(Base_Model).to(Device)
    Model.eval()

    # 2. Original-resolution prediction using the existing evaluator
    Evaluation = Evaluate_Model(Model=Model, Device=Device)
    Predictions = Evaluation['Predictions']

    Valid_Dataset, _ = Get_Validation_Dataloader()

    # 3. Locate the selected validation image
    Matched_Paths = [
        Path(Image_Path)
        for Image_Path in Predictions
        if Path(Image_Path).name == Target_Image_Name
    ]

    if len(Matched_Paths) != 1:
        raise ValueError(
            f'Expected one matching prediction for {Target_Image_Name}; '
            f'found {len(Matched_Paths)}'
        )

    Image_Path = Matched_Paths[0]
    Prediction_Mask = Predictions[str(Image_Path)].cpu().numpy()

    # 4. Load original image and ground-truth mask
    with Image.open(Image_Path) as Image_File:
        Original_Image = np.array(Image_File.convert('RGB'))

    Target_Mask = Valid_Dataset.Get_Original_Mask(str(Image_Path))
    if isinstance(Target_Mask, torch.Tensor):
        Target_Mask = Target_Mask.cpu().numpy()

    Target_Mask = np.asarray(Target_Mask)

    if Original_Image.shape[:2] != Target_Mask.shape:
        raise ValueError('Original image and target mask sizes differ.')

    if Prediction_Mask.shape != Target_Mask.shape:
        raise ValueError('Prediction and target mask sizes differ.')

    # 5. Convert class IDs to RGB colors
    Target_RGB = Class_Colors[Target_Mask.astype(np.int64)]
    Prediction_RGB = Class_Colors[Prediction_Mask.astype(np.int64)]

    # 6. Plot and save
    Figure, Axes = plt.subplots(1, 3, figsize=(15, 5))

    Panels = [
        (Original_Image, 'Original Image'),
        (Target_RGB, 'Ground Truth'),
        (Prediction_RGB, 'EXP-011_boundary Prediction'),
    ]

    for Axis, (Panel_Image, Panel_Title) in zip(Axes, Panels):
        Axis.imshow(Panel_Image)
        Axis.set_title(Panel_Title)
        Axis.axis('off')

    Figure.suptitle(Target_Image_Name, fontsize=11)
    Figure.tight_layout()

    Output_Path = Output_Dir / 'figure1_validation_comparison.png'
    Figure.savefig(
        Output_Path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    plt.close(Figure)

    print(f'Figure saved: {Output_Path}')
    print(f'Image size: {Original_Image.shape[1]} x {Original_Image.shape[0]}')
    print('Panels: Original Image / Ground Truth / Prediction')


if __name__ == '__main__':
    Main()