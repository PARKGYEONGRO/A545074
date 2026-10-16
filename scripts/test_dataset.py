# KoMaPDataSet 생성 및 첫 번째 데이터 추출 후 Tensor 형태, 해상도, Mask 클래스 값 검사 진행

import sys
from pathlib import Path

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
    TEST_IMAGE_DIR
)

from src.dataset import KoMaPDataset


# Dataset Test Function
def Test_Dataset(
    Dataset_Object,
    Dataset_Name,
    Is_Test=False
):

    print()
    print('=' * 60)
    print(f'{Dataset_Name} Dataset Test')
    print('=' * 60)

    # Dataset Length
    Dataset_Length = len(
        Dataset_Object
    )

    print(
        f'Dataset Length : {Dataset_Length}'
    )

    if Dataset_Length == 0:

        raise RuntimeError(
            f'{Dataset_Name} Dataset이 비어 있습니다.'
        )

    # First Sample
    Sample = Dataset_Object[0]

    Image_Tensor = Sample[
        'image'
    ]

    print(
        f'Image Shape    : {tuple(Image_Tensor.shape)}'
    )

    print(
        f'Image Dtype    : {Image_Tensor.dtype}'
    )

    print(
        f'Image Min      : {Image_Tensor.min().item():.4f}'
    )

    print(
        f'Image Max      : {Image_Tensor.max().item():.4f}'
    )

    print(
        f'Image Path     : {Sample["image_path"]}'
    )

    # Image Shape Check
    if not isinstance(
        Image_Tensor,
        torch.Tensor
    ):

        raise TypeError(
            'Image가 Tensor가 아닙니다.'
        )

    if Image_Tensor.ndim != 3:

        raise ValueError(
            f'Image Shape이 잘못되었습니다: '
            f'{tuple(Image_Tensor.shape)}'
        )

    if Image_Tensor.shape[0] != 3:

        raise ValueError(
            f'Image Channel이 3이 아닙니다: '
            f'{Image_Tensor.shape[0]}'
        )

    # Train / Valid
    if not Is_Test:

        Mask_Tensor = Sample[
            'mask'
        ]

        print(
            f'Mask Shape     : {tuple(Mask_Tensor.shape)}'
        )

        print(
            f'Mask Dtype     : {Mask_Tensor.dtype}'
        )

        Unique_Classes = torch.unique(
            Mask_Tensor
        ).tolist()

        print(
            f'Mask Classes   : {Unique_Classes}'
        )

        print(
            f'Mask Path      : {Sample["mask_path"]}'
        )

        # Mask Tensor Check
        if not isinstance(
            Mask_Tensor,
            torch.Tensor
        ):

            raise TypeError(
                'Mask가 Tensor가 아닙니다.'
            )

        if Mask_Tensor.ndim != 2:

            raise ValueError(
                f'Mask Shape이 잘못되었습니다: '
                f'{tuple(Mask_Tensor.shape)}'
            )

        # Mask Class Check
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

        # Image / Mask Resolution Check
        if (
            Image_Tensor.shape[1]
            != Mask_Tensor.shape[0]
            or
            Image_Tensor.shape[2]
            != Mask_Tensor.shape[1]
        ):

            raise ValueError(
                'Image와 Mask의 Tensor 해상도가 다릅니다.'
            )

    print()
    print(
        f'[PASS] {Dataset_Name} Dataset Test'
    )


# Main
def Main():

    print()
    print('=' * 60)
    print('KoMaP Dataset Test')
    print('=' * 60)

    # Train Dataset
    Train_Dataset = KoMaPDataset(
        Image_Dir=TRAIN_IMAGE_DIR,
        Mask_Dir=TRAIN_MASK_DIR,
        Is_Test=False
    )

    Test_Dataset(
        Dataset_Object=Train_Dataset,
        Dataset_Name='Train',
        Is_Test=False
    )

    # Valid Dataset
    Valid_Dataset = KoMaPDataset(
        Image_Dir=VALID_IMAGE_DIR,
        Mask_Dir=VALID_MASK_DIR,
        Is_Test=False
    )

    Test_Dataset(
        Dataset_Object=Valid_Dataset,
        Dataset_Name='Valid',
        Is_Test=False
    )

    # Test Dataset
    Test_Dataset_Object = KoMaPDataset(
        Image_Dir=TEST_IMAGE_DIR,
        Is_Test=True
    )

    Test_Dataset(
        Dataset_Object=Test_Dataset_Object,
        Dataset_Name='Test',
        Is_Test=True
    )

    # Final Result
    print()
    print('=' * 60)
    print('FINAL RESULT: PASS')
    print('모든 Dataset 기본 검사를 통과했습니다.')
    print('=' * 60)


# Entry Point
if __name__ == '__main__':

    Main()