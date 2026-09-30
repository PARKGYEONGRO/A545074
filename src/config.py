# Project 전체에서 사용할 경로, 클래스, Mask 색상, 학습 기본 설정 이곳에서 관리

from pathlib import Path


# 1. Project Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / 'data'

TRAIN_IMAGE_DIR = DATA_ROOT / 'train' / 'images'
TRAIN_MASK_DIR = DATA_ROOT / 'train' / 'masks'

VALID_IMAGE_DIR = DATA_ROOT / 'valid' / 'images'
VALID_MASK_DIR = DATA_ROOT / 'valid' / 'masks'

TEST_IMAGE_DIR = DATA_ROOT / 'test' / 'images'


# 2. Output Path
OUTPUT_ROOT = PROJECT_ROOT / 'outputs'

CHECKPOINT_DIR = OUTPUT_ROOT / 'checkpoints'
LOG_DIR = OUTPUT_ROOT / 'logs'
PREDICTION_DIR = OUTPUT_ROOT / 'predictions'


# 3. Class Configuration
CLASS_NAMES = [
    'Al',
    'Al3Ni',
    'Primary_Si',
    'Eutectic_Si'
]

NUM_CLASSES = len(CLASS_NAMES)


# 4. Class ID
CLASS_AL = 0
CLASS_AL3NI = 1
CLASS_PRIMARY_SI = 2
CLASS_EUTECTIC_SI = 3


# 5. Mask Color
MASK_COLOR_TO_CLASS = {
    (153, 127, 76): CLASS_AL,
    (25, 76, 153): CLASS_AL3NI,
    (76, 178, 76): CLASS_PRIMARY_SI,
    (204, 204, 204): CLASS_EUTECTIC_SI
}


# 6. Mask Class Color
CLASS_TO_MASK_COLOR = {
    CLASS_AL: (153, 127, 76),
    CLASS_AL3NI: (25, 76, 153),
    CLASS_PRIMARY_SI: (76, 178, 76),
    CLASS_EUTECTIC_SI: (204, 204, 204)
}


# 7. Patch Configuration
PATCH_SIZE = 384

PATCH_STRIDE = 192


# 8. Image Configuration
IMAGE_CHANNELS = 3


# 9. Training Configuration
RANDOM_SEED = 42

BATCH_SIZE = 8

NUM_WORKERS = 4

NUM_EPOCHS = 100


# 10. Optimization Configuration
LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4


# 11. Device Configuration
DEVICE = 'cuda'


# 12. Directory Creation
def Create_Directories():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )