from pathlib import Path
import hashlib
import re

from PIL import Image

from config import (
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VALID_IMAGE_DIR,
    VALID_MASK_DIR,
    TEST_IMAGE_DIR,
    MASK_COLOR_TO_CLASS
)


# 1. File Collection
def Get_Files(
    Directory,
    Pattern
):

    return sorted(
        Path(Directory).glob(Pattern)
    )


# 2. Filename Key
def Get_Sample_Key(
    File_Path
):

    return File_Path.stem.replace(
        '_image',
        ''
    ).replace(
        '_mask',
        ''
    )


# 3. SHA256
def Calculate_SHA256(
    File_Path
):

    Hash_Object = hashlib.sha256()

    with open(
        File_Path,
        'rb'
    ) as File:

        while True:

            Chunk = File.read(
                1024 * 1024
            )

            if not Chunk:
                break

            Hash_Object.update(
                Chunk
            )

    return Hash_Object.hexdigest()


# 4. A / Step / Sample Parsing
def Parse_Sample_Name(
    File_Path
):

    File_Name = File_Path.stem

    Match = re.match(
        r'(A\d+)_step(\d+)_sample(\d+)',
        File_Name
    )

    if Match is None:

        return None

    A_Number = Match.group(1)

    Step_Number = Match.group(2)

    Sample_Number = Match.group(3)

    return (
        A_Number,
        Step_Number,
        Sample_Number
    )


# 5. Basic File Check
def Check_File_Count():

    Train_Images = Get_Files(
        TRAIN_IMAGE_DIR,
        '*_image.png'
    )

    Train_Masks = Get_Files(
        TRAIN_MASK_DIR,
        '*_mask.png'
    )

    Valid_Images = Get_Files(
        VALID_IMAGE_DIR,
        '*_image.png'
    )

    Valid_Masks = Get_Files(
        VALID_MASK_DIR,
        '*_mask.png'
    )

    Test_Images = Get_Files(
        TEST_IMAGE_DIR,
        '*_image.png'
    )

    Test_Masks = Get_Files(
        TEST_IMAGE_DIR,
        '*_mask.png'
    )

    print(
        '\n[1] File Count'
    )

    print(
        f'    Train Image : {len(Train_Images)}'
    )

    print(
        f'    Train Mask  : {len(Train_Masks)}'
    )

    print(
        f'    Valid Image : {len(Valid_Images)}'
    )

    print(
        f'    Valid Mask  : {len(Valid_Masks)}'
    )

    print(
        f'    Test Image  : {len(Test_Images)}'
    )

    print(
        f'    Test Mask   : {len(Test_Masks)}'
    )

    if len(Train_Images) != len(Train_Masks):

        print(
            '    [FAIL] Train Image / Mask 개수가 다릅니다.'
        )

        return False

    if len(Valid_Images) != len(Valid_Masks):

        print(
            '    [FAIL] Valid Image / Mask 개수가 다릅니다.'
        )

        return False

    if len(Test_Masks) != 0:

        print(
            '    [WARN] Test에 Mask가 존재합니다.'
        )

    print(
        '    [PASS] File count'
    )

    return True


# 6. Filename Overlap
def Check_Filename_Overlap():

    Train_Images = Get_Files(
        TRAIN_IMAGE_DIR,
        '*_image.png'
    )

    Valid_Images = Get_Files(
        VALID_IMAGE_DIR,
        '*_image.png'
    )

    Test_Images = Get_Files(
        TEST_IMAGE_DIR,
        '*_image.png'
    )

    Train_Keys = {
        Get_Sample_Key(File)
        for File in Train_Images
    }

    Valid_Keys = {
        Get_Sample_Key(File)
        for File in Valid_Images
    }

    Test_Keys = {
        Get_Sample_Key(File)
        for File in Test_Images
    }

    Train_Valid = Train_Keys & Valid_Keys
    Train_Test = Train_Keys & Test_Keys
    Valid_Test = Valid_Keys & Test_Keys

    print(
        '\n[2] Filename Overlap'
    )

    if Train_Valid:

        print(
            f'    [FAIL] Train ↔ Valid: {sorted(Train_Valid)}'
        )

    else:

        print(
            '    [PASS] Train ↔ Valid'
        )

    if Train_Test:

        print(
            f'    [FAIL] Train ↔ Test: {sorted(Train_Test)}'
        )

    else:

        print(
            '    [PASS] Train ↔ Test'
        )

    if Valid_Test:

        print(
            f'    [FAIL] Valid ↔ Test: {sorted(Valid_Test)}'
        )

    else:

        print(
            '    [PASS] Valid ↔ Test'
        )

    return not (
        Train_Valid
        or Train_Test
        or Valid_Test
    )


# 7. Hash Duplicate Check
def Check_Image_Hash():

    Split_Files = {
        'Train': Get_Files(
            TRAIN_IMAGE_DIR,
            '*_image.png'
        ),
        'Valid': Get_Files(
            VALID_IMAGE_DIR,
            '*_image.png'
        ),
        'Test': Get_Files(
            TEST_IMAGE_DIR,
            '*_image.png'
        )
    }

    Hash_Map = {}

    for Split_Name, Files in Split_Files.items():

        for File_Path in Files:

            File_Hash = Calculate_SHA256(
                File_Path
            )

            if File_Hash not in Hash_Map:

                Hash_Map[
                    File_Hash
                ] = []

            Hash_Map[
                File_Hash
            ].append(
                (
                    Split_Name,
                    File_Path.name
                )
            )

    Duplicates = [
        Files
        for Files in Hash_Map.values()
        if len(Files) > 1
    ]

    print(
        '\n[3] Image Hash Duplicate'
    )

    if Duplicates:

        print(
            '    [FAIL] 동일 Image가 발견되었습니다.'
        )

        for Duplicate in Duplicates:

            print(
                f'    {Duplicate}'
            )

        return False

    print(
        '    [PASS] Image duplicate 없음'
    )

    return True


# 8. Mask Hash Duplicate Check
def Check_Mask_Hash():

    Split_Files = {
        'Train': Get_Files(
            TRAIN_IMAGE_DIR,
            '*_mask.png'
        ),
        'Valid': Get_Files(
            VALID_IMAGE_DIR,
            '*_mask.png'
        )
    }

    Hash_Map = {}

    for Split_Name, Files in Split_Files.items():

        for File_Path in Files:

            File_Hash = Calculate_SHA256(
                File_Path
            )

            if File_Hash not in Hash_Map:

                Hash_Map[
                    File_Hash
                ] = []

            Hash_Map[
                File_Hash
            ].append(
                (
                    Split_Name,
                    File_Path.name
                )
            )

    Duplicates = [
        Files
        for Files in Hash_Map.values()
        if len(Files) > 1
    ]

    print(
        '\n[4] Mask Hash Duplicate'
    )

    if Duplicates:

        print(
            '    [WARN] 동일 Mask가 발견되었습니다.'

        )

        for Duplicate in Duplicates:

            print(
                f'    {Duplicate}'
            )

        return False

    print(
        '    [PASS] Mask duplicate 없음'
    )

    return True


# 9. Image / Mask Matching
def Check_Image_Mask_Matching():

    print(
        '\n[5] Image / Mask Matching'
    )

    Is_Passed = True

    for Split_Name, Image_Directory, Mask_Directory in [
        ('Train', TRAIN_IMAGE_DIR, TRAIN_MASK_DIR),
        ('Valid', VALID_IMAGE_DIR, VALID_MASK_DIR)
    ]:

        Image_Files = Get_Files(
            Image_Directory,
            '*_image.png'
        )

        Mask_Files = Get_Files(
            Mask_Directory,
            '*_mask.png'
        )

        Image_Keys = {
            Get_Sample_Key(File)
            for File in Image_Files
        }

        Mask_Keys = {
            Get_Sample_Key(File)
            for File in Mask_Files
        }

        Missing_Mask = (
            Image_Keys - Mask_Keys
        )

        Extra_Mask = (
            Mask_Keys - Image_Keys
        )

        if Missing_Mask:

            print(
                f'    [FAIL] {Split_Name} Mask 없음: '
                f'{sorted(Missing_Mask)}'
            )

            Is_Passed = False

        if Extra_Mask:

            print(
                f'    [FAIL] {Split_Name} Image 없음: '
                f'{sorted(Extra_Mask)}'
            )

            Is_Passed = False

    if Is_Passed:

        print(
            '    [PASS] 모든 Image ↔ Mask 매칭 정상'
        )

    return Is_Passed


# 10. Resolution Check
def Check_Resolution():

    print(
        '\n[6] Image / Mask Resolution'
    )

    Is_Passed = True

    for Split_Name, Image_Directory, Mask_Directory in [
        ('Train', TRAIN_IMAGE_DIR, TRAIN_MASK_DIR),
        ('Valid', VALID_IMAGE_DIR, VALID_MASK_DIR)
    ]:

        Image_Files = Get_Files(
            Image_Directory,
            '*_image.png'
        )

        for Image_Path in Image_Files:

            Sample_Key = Get_Sample_Key(
                Image_Path
            )

            Mask_Path = (
                Mask_Directory /
                f'{Sample_Key}_mask.png'
            )

            with Image.open(
                Image_Path
            ) as Image_File:

                Image_Size = Image_File.size

            with Image.open(
                Mask_Path
            ) as Mask_File:

                Mask_Size = Mask_File.size

            if Image_Size != Mask_Size:

                print(
                    f'    [FAIL] {Split_Name}: '
                    f'{Image_Path.name}'
                )

                print(
                    f'           Image={Image_Size}, '
                    f'Mask={Mask_Size}'
                )

                Is_Passed = False

    if Is_Passed:

        print(
            '    [PASS] 모든 Image / Mask 해상도 일치'
        )

    return Is_Passed


# 11. Mask Color Check
def Check_Mask_Color():

    print(
        '\n[7] Mask Color'
    )

    Is_Passed = True

    Valid_Colors = set(
        MASK_COLOR_TO_CLASS.keys()
    )

    for Split_Name, Mask_Directory in [
        ('Train', TRAIN_MASK_DIR),
        ('Valid', VALID_MASK_DIR)
    ]:

        Mask_Files = Get_Files(
            Mask_Directory,
            '*_mask.png'
        )

        for Mask_Path in Mask_Files:

            with Image.open(
                Mask_Path
            ) as Mask:

                Mask_RGB = Mask.convert(
                    'RGB'
                )

                Colors = set(
                    Mask_RGB.getdata()
                )

            Unknown_Colors = (
                Colors - Valid_Colors
            )

            if Unknown_Colors:

                print(
                    f'    [FAIL] {Split_Name}: '
                    f'{Mask_Path.name}'
                )

                print(
                    f'           Unknown colors: '
                    f'{sorted(Unknown_Colors)}'
                )

                Is_Passed = False

    if Is_Passed:

        print(
            '    [PASS] 모든 Mask 색상 정상'
        )

    return Is_Passed


# 12. A Group Overlap
def Check_A_Group_Overlap():

    Split_Files = {
        'Train': Get_Files(
            TRAIN_IMAGE_DIR,
            '*_image.png'
        ),
        'Valid': Get_Files(
            VALID_IMAGE_DIR,
            '*_image.png'
        ),
        'Test': Get_Files(
            TEST_IMAGE_DIR,
            '*_image.png'
        )
    }

    Split_Groups = {}

    for Split_Name, Files in Split_Files.items():

        Split_Groups[
            Split_Name
        ] = {}

        for File_Path in Files:

            Parsed = Parse_Sample_Name(
                File_Path
            )

            if Parsed is None:
                continue

            A_Number = Parsed[0]

            if A_Number not in Split_Groups[
                Split_Name
            ]:

                Split_Groups[
                    Split_Name
                ][A_Number] = []

            Split_Groups[
                Split_Name
            ][A_Number].append(
                File_Path.name
            )

    Train_A = set(
        Split_Groups['Train'].keys()
    )

    Valid_A = set(
        Split_Groups['Valid'].keys()
    )

    Test_A = set(
        Split_Groups['Test'].keys()
    )

    Train_Valid = Train_A & Valid_A
    Train_Test = Train_A & Test_A
    Valid_Test = Valid_A & Test_A

    print(
        '\n[8] A Group Overlap'
    )

    if Train_Valid:

        print(
            f'    [WARN] Train ↔ Valid A overlap: '
            f'{sorted(Train_Valid)}'
        )

    else:

        print(
            '    [PASS] Train ↔ Valid A overlap 없음'
        )

    if Train_Test:

        print(
            f'    [WARN] Train ↔ Test A overlap: '
            f'{sorted(Train_Test)}'
        )

    else:

        print(
            '    [PASS] Train ↔ Test A overlap 없음'
        )

    if Valid_Test:

        print(
            f'    [WARN] Valid ↔ Test A overlap: '
            f'{sorted(Valid_Test)}'
        )

    else:

        print(
            '    [PASS] Valid ↔ Test A overlap 없음'
        )

    print(
        '    ※ A 번호의 공식적인 의미가 정의되지 않았으므로 '
        '자동 재분할하지 않습니다.'
    )


# 13. Main
def Main():

    print(
        '=' * 60
    )

    print(
        'KoMaP Dataset Leakage Check'
    )

    print(
        '=' * 60
    )

    Results = []

    Results.append(
        Check_File_Count()
    )

    Results.append(
        Check_Filename_Overlap()
    )

    Results.append(
        Check_Image_Hash()
    )

    Results.append(
        Check_Mask_Hash()
    )

    Results.append(
        Check_Image_Mask_Matching()
    )

    Results.append(
        Check_Resolution()
    )

    Results.append(
        Check_Mask_Color()
    )

    Check_A_Group_Overlap()

    print(
        '\n' + '=' * 60
    )

    if all(Results):

        print(
            'FINAL RESULT: PASS'
        )

        print(
            '직접 확인 가능한 데이터 무결성 검사를 통과했습니다.'
        )

    else:

        print(
            'FINAL RESULT: FAIL'
        )

        print(
            '문제를 수정한 후 학습을 진행해야 합니다.'
        )

    print(
        '=' * 60
    )


if __name__ == '__main__':

    Main()