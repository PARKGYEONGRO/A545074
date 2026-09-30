# 이미지와 mask가 함께 변환되고, mask의 클래스 값이 깨지지 않는지 확인

import sys
from pathlib import Path

import numpy as np
from PIL import Image


# 1. Project Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))


# 2. Import
from src.config import (
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR
)

from src.transforms import (
    Get_Train_Transform,
    Get_Valid_Transform,
    Get_Test_Transform
)


# 3. Transform Test
def Test_Transform():
    print('=' * 60)
    print('KoMaP Transform Test')
    print('=' * 60)

    # Image ↔ Mask Matching
    Image_Paths = sorted(
        TRAIN_IMAGE_DIR.glob('*.png')
    )

    Mask_Paths = sorted(
        TRAIN_MASK_DIR.glob('*.png')
    )

    assert len(Image_Paths) > 0, (
        'Train Image 파일이 없습니다.'
    )

    assert len(Mask_Paths) > 0, (
        'Train Mask 파일이 없습니다.'
    )

    Image_Path = Image_Paths[0]


    # Image ↔ Mask Matching
    Image_Paths = sorted(
        TRAIN_IMAGE_DIR.glob('*.png')
    )

    assert len(Image_Paths) > 0, (
        'Train Image 파일이 없습니다.'
    )

    Image_Path = Image_Paths[0]

    Mask_File_Name = Image_Path.name.replace(
        '_image.png',
        '_mask.png'
    )

    Mask_Path = TRAIN_MASK_DIR / Mask_File_Name

    assert Mask_Path.exists(), (
        f'Mask 파일을 찾을 수 없습니다: {Mask_Path}'
    )

    assert Mask_Path.exists(), (
        f'Mask 파일을 찾을 수 없습니다: {Mask_Path}'
    )

    Image_File = Image.open(
        Image_Path
    ).convert('RGB')

    Mask_File = Image.open(
        Mask_Path
    ).convert('RGB')

    Image_Array = np.asarray(
        Image_File
    )

    Mask_Array = np.asarray(
        Mask_File
    )

    print()
    print('[Original]')
    print(f'Image Shape : {Image_Array.shape}')
    print(f'Mask Shape  : {Mask_Array.shape}')

    # Train Transform
    Train_Transform = Get_Train_Transform()

    Train_Result = Train_Transform(
        image=Image_Array,
        mask=Mask_Array
    )

    Train_Image = Train_Result['image']
    Train_Mask = Train_Result['mask']

    print()
    print('[Train Transform]')
    print(f'Image Shape : {Train_Image.shape}')
    print(f'Mask Shape  : {Train_Mask.shape}')

    assert Train_Image.shape[:2] == Train_Mask.shape[:2], (
        'Train Transform 후 Image와 Mask의 크기가 다릅니다.'
    )

    # Mask Color Test
    Original_Colors = set(
        map(
            tuple,
            Mask_Array.reshape(-1, 3)
        )
    )

    Train_Colors = set(
        map(
            tuple,
            Train_Mask.reshape(-1, 3)
        )
    )

    print()
    print('[Mask Color Test]')
    print(f'Original Colors : {sorted(Original_Colors)}')
    print(f'Train Colors    : {sorted(Train_Colors)}')

    assert Train_Colors.issubset(
        Original_Colors
    ), (
        'Transform 과정에서 새로운 Mask 색상이 생성되었습니다.'
    )

    # Valid Transform
    Valid_Transform = Get_Valid_Transform()

    Valid_Result = Valid_Transform(
        image=Image_Array,
        mask=Mask_Array
    )

    Valid_Image = Valid_Result['image']
    Valid_Mask = Valid_Result['mask']

    print()
    print('[Valid Transform]')
    print(f'Image Shape : {Valid_Image.shape}')
    print(f'Mask Shape  : {Valid_Mask.shape}')

    assert np.array_equal(
        Image_Array,
        Valid_Image
    ), (
        'Valid Transform에서 Image가 변경되었습니다.'
    )

    assert np.array_equal(
        Mask_Array,
        Valid_Mask
    ), (
        'Valid Transform에서 Mask가 변경되었습니다.'
    )

    print()
    print('[PASS] Valid Transform Test')

    # Test Transform
    Test_Transform = Get_Test_Transform()

    Test_Result = Test_Transform(
        image=Image_Array
    )

    Test_Image = Test_Result['image']

    print()
    print('[Test Transform]')
    print(f'Image Shape : {Test_Image.shape}')

    assert np.array_equal(
        Image_Array,
        Test_Image
    ), (
        'Test Transform에서 Image가 변경되었습니다.'
    )

    print()
    print('[PASS] Test Transform Test')

    # Final Result
    print()
    print('=' * 60)
    print('FINAL RESULT: PASS')
    print('모든 Transform 검사를 통과했습니다.')
    print('=' * 60)


# 4. Main
if __name__ == '__main__':
    Test_Transform()