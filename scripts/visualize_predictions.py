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
from src.evaluate import(
    Get_Validation_Dataloader,
    Reconstruct_Original_Preditions
)


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

# 4. Main
def Main():
    Parser = argparse.ArgumentParser()

    Parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Optional trained model checkpoint path.'
    )

    Parser.add_argument(
        '--limit',
        type=int,
        default=5,
        help='Number of validation images to visualize.'
    )

    Args = Parser.parse_args()

    if Args.limit < 1:
        raise ValueError('--limit must be at least 1.')

    Device = torch.device(DEVICE)

    Model = Get_Model().to(Device)

    # Load Checkpoint If Provided
    if Args.checkpoint:
        Checkpoint_Path = Path(Args.checkpoint)

        if not Checkpoint_Path.is_file():
            raise FileNotFoundError(f'Check not found: {Checkpoint_Path}')

        Checkpoint = torch.load(Checkpoint_Path, map_location=Device)

        if (isinstance(Checkpoint, dict) and 'model_state_dict' in Checkpoint):
            State_Dict = Checkpoint['model_state_dict']
        else:
            State_Dict = Checkpoint

        Model.load_state_dict(State_Dict)


        print(f'Loaded checkpoint: {Checkpoint_Path}')

    else:
        print('WARNING: No checkpoint supplied.\nVisualizing an untrained model.')


    # Reconstruct Full-Resolution Predictions
    Valid_Dataset, Valid_Dataloader = (
        Get_Validation_Dataloader()
    )

    Original_Predicitons = (
        Reconstruct_Original_Preditions(
            Model=Model,
            Valid_Dataset=Valid_Dataset,
            Valid_Dataloader=Valid_Dataloader,
            Device=Device
        )
    )

    Output_Dir = (PROJECT_ROOT/'outputs'/'visualizations')

    Saved_Count = 0

    for Image_Path, Predictions_Mask in (Original_Predicitons.items()):
        Target_Mask = (Valid_Dataset.Get_Original_Mask(Image_Path))

        Output_path = Save_Visualization(
            Image_Path=Image_Path,
            Target_Mask=Target_Mask,
            Prediction_Mask=Predictions_Mask,
            Output_Dir=Output_Dir
        )

        print(f'Save: {Output_path}')

        Saved_Count += 1

        if Saved_Count >= Args.limit:
            break

    print('=' * 60)
    print('Visualization completed.')
    print(f'Saved images: {Saved_Count}')
    print(f'Output directory: {Output_Dir}')
    print('=' * 60)


if __name__ == '__main__':
    Main()