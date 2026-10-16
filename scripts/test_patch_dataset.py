# =========================================================
# Patch Dataset Test
# =========================================================

import sys
from pathlib import Path
from PIL import Image

import torch


# 1. Project Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)


# 2. Import
from src.config import (
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VALID_IMAGE_DIR,
    VALID_MASK_DIR,
    TEST_IMAGE_DIR,
    PATCH_SIZE
)

from src.patch_dataset import KoMaPPatchDataset


# Patch Sample Test
def Test_Patch_Sample(
    Dataset_Object,
    Dataset_Name,
    Is_Test=False
):

    print()
    print('=' * 60)
    print(f'{Dataset_Name} Patch Test')
    print('=' * 60)

    Dataset_Length = len(
        Dataset_Object
    )

    print(
        f'Patch Count    : {Dataset_Length}'
    )

    if Dataset_Length == 0:

        raise RuntimeError(
            f'{Dataset_Name} Patch가 생성되지 않았습니다.'
        )


    # Sample Check
    Sample_Indices = [
        0,
        Dataset_Length // 2,
        Dataset_Length - 1
    ]

    Sample_Indices = sorted(
        set(Sample_Indices)
    )

    for Sample_Index in Sample_Indices:

        Sample = Dataset_Object[
            Sample_Index
        ]

        Image_Tensor = Sample[
            'image'
        ]

        X = Sample[
            'x'
        ]

        Y = Sample[
            'y'
        ]

        print()
        print(
            f'[Sample {Sample_Index}]'
        )

        print(
            f'Image Shape    : {tuple(Image_Tensor.shape)}'
        )

        print(
            f'X              : {X}'
        )

        print(
            f'Y              : {Y}'
        )


        # Image Tensor Check
        if not isinstance(
            Image_Tensor,
            torch.Tensor
        ):

            raise TypeError(
                'Image가 Tensor가 아닙니다.'
            )

        if Image_Tensor.shape != (
            3,
            PATCH_SIZE,
            PATCH_SIZE
        ):

            raise ValueError(
                'Patch Image Shape이 잘못되었습니다: '
                f'{tuple(Image_Tensor.shape)}'
            )


        # Train / Valid Mask Check
        if not Is_Test:

            Mask_Tensor = Sample[
                'mask'
            ]

            Unique_Classes = torch.unique(
                Mask_Tensor
            ).tolist()

            print(
                f'Mask Shape     : '
                f'{tuple(Mask_Tensor.shape)}'
            )

            print(
                f'Mask Classes   : '
                f'{Unique_Classes}'
            )

            if Mask_Tensor.shape != (
                PATCH_SIZE,
                PATCH_SIZE
            ):

                raise ValueError(
                    'Patch Mask Shape이 잘못되었습니다: '
                    f'{tuple(Mask_Tensor.shape)}'
                )

            Invalid_Classes = [
                Class_ID
                for Class_ID in Unique_Classes
                if Class_ID < 0 or Class_ID > 3
            ]

            if Invalid_Classes:

                raise ValueError(
                    f'잘못된 Mask Class가 있습니다: '
                    f'{Invalid_Classes}'
                )

    print()
    print(
        f'[PASS] {Dataset_Name} Patch Sample Test'
    )



# Patch Coordinate Coverage Test
def Test_Patch_Coverage(
    Dataset_Object,
    Dataset_Name
):

    print()
    print('=' * 60)
    print(f'{Dataset_Name} Patch Coverage Test')
    print('=' * 60)

    Image_Coverage = {}

    for (
        Image_Index,
        X,
        Y
    ) in Dataset_Object.Patch_Indices:

        if Image_Index not in Image_Coverage:

            Image_Coverage[
                Image_Index
            ] = []

        Image_Coverage[
            Image_Index
        ].append(
            (
                X,
                Y
            )
        )

    for Image_Index in Image_Coverage:

        Image_Path = Dataset_Object.Base_Dataset.Image_Paths[
            Image_Index
        ]

        from PIL import Image

        with Image.open(
            Image_Path
        ) as Image_File:

            Width, Height = Image_File.size

        Patch_Coordinates = Image_Coverage[
            Image_Index
        ]

        X_Positions = [
            Coordinate[0]
            for Coordinate in Patch_Coordinates
        ]

        Y_Positions = [
            Coordinate[1]
            for Coordinate in Patch_Coordinates
        ]


        # Right Edge Check
        Right_Edge = max(
            X_Positions
        ) + PATCH_SIZE

        if Right_Edge != Width:

            raise ValueError(
                'Image 오른쪽 끝이 Patch로 '
                '커버되지 않았습니다.\n'
                f'Image : {Image_Path}\n'
                f'Width : {Width}\n'
                f'Right Edge : {Right_Edge}'
            )


        # Bottom Edge Check
        Bottom_Edge = max(
            Y_Positions
        ) + PATCH_SIZE

        if Bottom_Edge != Height:

            raise ValueError(
                'Image 아래쪽 끝이 Patch로 '
                '커버되지 않았습니다.\n'
                f'Image : {Image_Path}\n'
                f'Height : {Height}\n'
                f'Bottom Edge : {Bottom_Edge}'
            )


        # Coordinate Range Check
        for X, Y in Patch_Coordinates:

            if X < 0 or Y < 0:

                raise ValueError(
                    f'잘못된 Patch 좌표가 있습니다: '
                    f'({X}, {Y})'
                )

            if X + PATCH_SIZE > Width:

                raise ValueError(
                    f'Patch가 오른쪽 범위를 '
                    f'벗어났습니다: ({X}, {Y})'
                )

            if Y + PATCH_SIZE > Height:

                raise ValueError(
                    f'Patch가 아래쪽 범위를 '
                    f'벗어났습니다: ({X}, {Y})'
                )

    print(
        f'Image Count    : {len(Image_Coverage)}'
    )

    print()
    print(
        f'[PASS] {Dataset_Name} Patch Coverage Test'
    )


# Main
def Main():

    print()
    print('=' * 60)
    print('KoMaP Patch Dataset Test')
    print('=' * 60)


    # Train Dataset
    Train_Dataset = KoMaPPatchDataset(
        Image_Dir=TRAIN_IMAGE_DIR,
        Mask_Dir=TRAIN_MASK_DIR,
        Is_Test=False
    )

    Test_Patch_Sample(
        Dataset_Object=Train_Dataset,
        Dataset_Name='Train',
        Is_Test=False
    )

    Test_Patch_Coverage(
        Dataset_Object=Train_Dataset,
        Dataset_Name='Train'
    )


    # Valid Dataset
    Valid_Dataset = KoMaPPatchDataset(
        Image_Dir=VALID_IMAGE_DIR,
        Mask_Dir=VALID_MASK_DIR,
        Transform=None,
        Is_Test=False
    )

    Test_Patch_Sample(
        Dataset_Object=Valid_Dataset,
        Dataset_Name='Valid',
        Is_Test=False
    )

    Test_Patch_Coverage(
        Dataset_Object=Valid_Dataset,
        Dataset_Name='Valid'
    )


    # Test Dataset
    Test_Dataset = KoMaPPatchDataset(
        Image_Dir=TEST_IMAGE_DIR,
        Is_Test=True
    )

    Test_Patch_Sample(
        Dataset_Object=Test_Dataset,
        Dataset_Name='Test',
        Is_Test=True
    )

    Test_Patch_Coverage(
        Dataset_Object=Test_Dataset,
        Dataset_Name='Test'
    )

    Sample = Valid_Dataset[0]

    Original_Mask = Valid_Dataset.Get_Original_Mask(Sample['image_path'])

    with Image.open(Sample['image_path']) as Image_File:

        Original_Width, Original_Height = Image_File.size

    print('\n[Original Mask Test]')
    print('Image Path     :', Sample['image_path'])
    print('Original Shape :', tuple(Original_Mask.shape))
    print('Mask Dtype     :', Original_Mask.dtype)
    print('Mask Classes   :', torch.unique(Original_Mask).tolist())
    assert Original_Mask.ndim == 2
    assert Original_Mask.shape == (Original_Height, Original_Width), '원본 이미지와 정답 마스크의 크기가 다릅니다.'
    assert Original_Mask.dtype == torch.int64
    assert set(torch.unique(Original_Mask).tolist()).issubset({0, 1, 2, 3})

    print('[PASS] Original Mask Retrieval Test')

    # Final Result
    print()
    print('=' * 60)
    print('FINAL RESULT: PASS')
    print('모든 Patch Dataset 검사를 통과했습니다.')
    print('=' * 60)


# Entry Point
if __name__ == '__main__':

    Main()