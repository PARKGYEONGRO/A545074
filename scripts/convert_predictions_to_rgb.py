from pathlib import Path

import numpy as np
from PIL import Image

from src.config import (
    NUM_CLASSES,
    CLASS_TO_MASK_COLOR
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = (PROJECT_ROOT / 'outputs' / 'final' / 'test_predictions')

OUTPUT_DIR = (PROJECT_ROOT / 'outputs' / 'final' / 'test_predictions_rgb')


def Convert_Mask_To_RGB(Mask_Array):

    if Mask_Array.ndim != 2:
        raise ValueError(f'Expected a 2D class-ID mask, got {Mask_Array.shape}')

    Unique_Classes = np.unique(Mask_Array)

    if not np.isin(Unique_Classes, np.arange(NUM_CLASSES)).all():
        raise ValueError(f'Invalid class IDs: {Unique_Classes.tolist()}')

    RGB_Mask = np.zeros((*Mask_Array.shape, 3), dtype=np.uint8)

    for Class_ID, RGB_Color in CLASS_TO_MASK_COLOR.items():

        RGB_Mask[Mask_Array == Class_ID] = RGB_Color

    return RGB_Mask


def Main():

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f'Prediction directory not found: {INPUT_DIR}')

    Input_Files = sorted(INPUT_DIR.glob('*_mask.png'))

    if len(Input_Files) != 10:
        raise RuntimeError(f'Expected 10 predictions, found {len(Input_Files)}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for Input_Path in Input_Files:

        with Image.open(Input_Path) as Image_File:
            Mask_Array = np.array(Image_File)

        RGB_Mask = Convert_Mask_To_RGB(Mask_Array)

        Output_Path = OUTPUT_DIR / Input_Path.name

        Image.fromarray(RGB_Mask, mode='RGB').save(Output_Path)

        with Image.open(Output_Path) as Saved_Image:
            if Saved_Image.mode != 'RGB':
                raise RuntimeError(f'Unexpected output mode: {Output_Path}')

            if Saved_Image.size != (Mask_Array.shape[1], Mask_Array.shape[0]):
                raise RuntimeError(f'Output resolution mismatch: {Output_Path}')

        print(
            f'{Input_Path.name} -> {Output_Path.name} | '
            f'shape={RGB_Mask.shape}'
        )

    Output_Files = list(OUTPUT_DIR.glob('*_mask.png'))

    if len(Output_Files) != 10:
        raise RuntimeError(f'Output count mismatch: {len(Output_Files)}')

    print('-' * 60)
    print(f'RGB masks created: {len(Output_Files)}')
    print(f'Output directory: {OUTPUT_DIR}')
    print('RGB conversion validation: PASS')


if __name__ == '__main__':
    Main()