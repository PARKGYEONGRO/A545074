# check_class_distribution.py 각 클래스 별 전체 픽셀에 차지하는 비율 확인
from collections import Counter
from pathlib import Path

import sys
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)

from src.config import(
    CLASS_NAMES,
    MASK_COLOR_TO_CLASS,
    TRAIN_MASK_DIR,
    VALID_MASK_DIR,
    NUM_CLASSES
)

# 1. Mask Class Distribution
def Analyze_Mask_Distribution(Mask_Dir: Path):
    Mask_Paths = sorted(Mask_Dir.glob('*_mask.png'))

    if len(Mask_Paths) == 0:
        raise RuntimeError(f'Mask를 찾을 수 없습니다: {Mask_Dir}')

    Pixel_Counts = Counter()
    Total_Pixels = 0

    for Mask_Path in Mask_Paths:

        Mask_RGB = np.array(Image.open(Mask_Path).convert('RGB'))

        Encoded_Mask = np.full(
            Mask_RGB.shape[:2],
            -1,
            dtype=np.int64
        )

        for Color, Class_ID in MASK_COLOR_TO_CLASS.items():
            Match = np.all(Mask_RGB == Color, axis=-1)

            Encoded_Mask[Match] = Class_ID

        Unknown_Pixels = Encoded_Mask == -1

        if np.any(Unknown_Pixels):
            Unknown_Colors = np.unique(Mask_RGB[Unknown_Pixels], axis=0)

            raise ValueError(
                f'알 수 없는 색상: {Mask_Path}\n'
                f'{Unknown_Colors.tolist()}'
            )
        
        Counts = np.bincount(Encoded_Mask.ravel(), minlength=NUM_CLASSES)

        Pixel_Counts.update({
            Class_ID: int(Counts[Class_ID])
            for Class_ID in range(NUM_CLASSES)
        })

        Total_Pixels += Encoded_Mask.size

    print(f'\nMask Directory: {Mask_Dir}')
    print(f'Image Count: {len(Mask_Paths)}')
    print(f'Total Pixels: {Total_Pixels:,}')

    for Class_ID, Class_Name in enumerate(CLASS_NAMES):
        Count = Pixel_Counts[Class_ID]
        Ratio = Count / Total_Pixels * 100

        print(
            f'{Class_ID} | {Class_Name:<15} | '
            f'{Count:>12,} Pixels | {Ratio:>8.3f}'
        )


# 2. Main
if __name__ == '__main__':
    Analyze_Mask_Distribution(TRAIN_MASK_DIR)
    Analyze_Mask_Distribution(VALID_MASK_DIR)