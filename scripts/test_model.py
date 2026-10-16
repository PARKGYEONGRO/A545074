import sys
from pathlib import Path

import torch




# 1. Project Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project Import
from src.config import(
    NUM_CLASSES,
    PATCH_SIZE
)

from src.model import Get_Model

# 2. Model Test Configuration
BATCH_SIZE = 1
IMAGE_CHANNELS = 3

EXPECTED_OUTPUT_SHAPE = (
    BATCH_SIZE,
    NUM_CLASSES,
    PATCH_SIZE,
    PATCH_SIZE
)


# 3. Device
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


# 4. Model Test
def Test_Model():
    print ('====================\nKoMaP Model Test\n====================')
    print()

    print(f'[Device]: {DEVICE}')

    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('GPU: Not Available')
    print()

    print('[Model Creation]')
    Model = Get_Model()
    Model = Model.to(DEVICE)
    Model.eval()

    print('Model           : U-Net\n' \
    'Encoder         : ResNet34\n' \
    'Encoder Weights : ImageNet')
    print()

    # Parameter Count
    Total_Parameters = sum(
        Parameter.numel() for Parameter in Model.parameters()
    )

    Trainable_Parameters = sum(
        Parameter.numel() for Parameter in Model.parameters()
        if Parameter.requires_grad
    )

    print('[Parameter Count]')
    print(f'Total Parameters     : {Total_Parameters:,}\n'
          f'Trainable Parameters : {Trainable_Parameters:,}')
    print()

    # Dummy Input
    Image = torch.randn(
        BATCH_SIZE,
        IMAGE_CHANNELS,
        PATCH_SIZE,
        PATCH_SIZE,
        device=DEVICE
    )

    print('[Input]')
    print(f'Input Shape: {tuple(Image.shape)}\n'
          f'Input Dtype: {Image.dtype}')
    print()

    # Forward
    print('[Forward Test]')

    try:
        with torch.no_grad():
            Output = Model(Image)

    except Exception as Error:
        print(f'[Fail] Forward Error\nError: {Error}')

        return False

    # Output Check
    print(f'Output Shape: {tuple(Output.shape)}\n'
          f'Output Dtype: {Output.dtype}')
    print()

    Shape_Test = tuple(Output.shape) == EXPECTED_OUTPUT_SHAPE
    Dtype_Test = Output.dtype == torch.float32

    print('[Shape Test]')
    if Shape_Test:
        print(f'[PASS] Output Shape : {tuple(Output.shape)}')
    else:
        print(f'[FAIL] Expected     : {EXPECTED_OUTPUT_SHAPE}')
    print()

    print('[Dtype Test]')
    if Dtype_Test:
        print('[PASS] Output Dtype: torch.float32')
    else:
        print(f'[FAIL] Output Dtype: {Output.dtype}')
    print()

    # Final Result
    Final_Result = Shape_Test and Dtype_Test
    print ('====================\nFINAL RESULT')
    if Final_Result:
        print('[PASS]\n====================')
    else:
        print('[FAIL]\n====================')

    return Final_Result


# 5. Main
if __name__ == '__main__':
    Result = Test_Model()

    if not Result:
        sys.exit(1)
    