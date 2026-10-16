
# scripts/train_boundary.py
# 경계 인식 다중 과제 학습

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VALID_IMAGE_DIR,
    VALID_MASK_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    RANDOM_SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    WEIGHT_DECAY,
    DEVICE,
    NUM_CLASSES
)

from src.patch_dataset import KoMaPPatchDataset
from src.transforms import (
    Get_Train_Transform,
    Get_Valid_Transform
)

from src.boundary_model import Get_Boundary_Model
from src.losses import Get_Loss

from src.evaluate import (
    Get_Original_Image_Size,
    Evaluate_Original_Images
)


# 1. Experiment Configuration
EXPERIMENT_NAME = 'exp015_boundary_finetune_exp013'

INIT_CHECKPOINT = (CHECKPOINT_DIR / 'exp013_finetune_exp009' / 'best_model.pth')

EXPERIMENT_CHECKPOINT_DIR = (CHECKPOINT_DIR / EXPERIMENT_NAME)

EXPERIMENT_LOG_DIR = (LOG_DIR / EXPERIMENT_NAME)

BOUNDARY_WEIGHT = 0.2
DICE_WEIGHT = 0.25
DEFAULT_LEARNING_RATE = 5e-6
DEFAULT_EPOCHS = 5

EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.0001


# 2. Reproducibility and Device
def Set_Seed(Seed=RANDOM_SEED):

    random.seed(Seed)
    np.random.seed(Seed)
    torch.manual_seed(Seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def Set_Device():
    if DEVICE == 'cuda' and torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    print(f'Using Device: {Device}')

    if Device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(Device)}')

    return Device


# 3. Dataset and Dataloader
def Get_Train_Dataloader(Device):

    Train_Dataset = KoMaPPatchDataset(
        Image_Dir=TRAIN_IMAGE_DIR,
        Mask_Dir=TRAIN_MASK_DIR,
        Transform=Get_Train_Transform(),
        Is_Test=False
    )

    Generator = torch.Generator()
    Generator.manual_seed(RANDOM_SEED)

    Train_Dataloader = DataLoader(
        Train_Dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(Device.type == 'cuda'),
        generator=Generator,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False
    )

    if len(Train_Dataset) == 0:
        raise RuntimeError('Training Dataset is empty')

    print(f'Train Images: {len(Train_Dataset.Base_Dataset)}')
    print(f'Train Patches: {len(Train_Dataset)}')

    return Train_Dataset, Train_Dataloader


def Get_Valid_Dataloader(Device):
    Valid_Dataset = KoMaPPatchDataset(
        Image_Dir=VALID_IMAGE_DIR,
        Mask_Dir=VALID_MASK_DIR,
        Transform=Get_Valid_Transform(),
        Is_Test=False
    )

    Valid_Dataloader = DataLoader(
        Valid_Dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(Device.type == 'cuda'),
        persistent_workers=(NUM_WORKERS > 0)
    )

    return Valid_Dataset, Valid_Dataloader


# 4. Boundary Target Generation
def Generate_Boundary_Target(Masks):

    # Masks: [B, H, W]
    Mask_Float = Masks.float().unsqueeze(1)

    # 3x3 주변의 클래스 ID 최댓값과 최솟값 비교
    Local_Max = F.max_pool2d(Mask_Float, kernel_size=3, stride=1, padding=1)

    Local_Min = -F.max_pool2d(-Mask_Float, kernel_size=3, stride=1, padding=1)

    Boundary_Target = (Local_Max != Local_Min).float()

    return Boundary_Target



# 5. Checkpoint Loading
def Load_Initial_Checkpoint(Model, Device):

    if not INIT_CHECKPOINT.exists():
        raise FileNotFoundError(
            f'Initial checkpoint not found: {INIT_CHECKPOINT}'
        )

    Checkpoint = torch.load(
        INIT_CHECKPOINT,
        map_location=Device,
        weights_only=False
    )

    if 'model_state_dict' not in Checkpoint:
        raise KeyError(
            'model_state_dict not found in initial checkpoint'
        )

    Initial_State_Dict = Checkpoint['model_state_dict']
    Boundary_State_Dict = Model.state_dict()

    Transferred_State_Dict = {}
    Unexpected_Keys = []
    Shape_Mismatch_Keys = []

    for Key, Value in Initial_State_Dict.items():

        if Key.startswith('Model.'):
            Boundary_Key = Key
        else:
            Boundary_Key = f'Model.{Key}'

        if Boundary_Key not in Boundary_State_Dict:
            Unexpected_Keys.append(Key)
            continue

        if Boundary_State_Dict[Boundary_Key].shape != Value.shape:
            Shape_Mismatch_Keys.append(
                (
                    Key,
                    tuple(Value.shape),
                    tuple(Boundary_State_Dict[Boundary_Key].shape)
                )
            )
            continue

        Transferred_State_Dict[Boundary_Key] = Value

    if Unexpected_Keys:
        raise RuntimeError(
            f'Unexpected checkpoint keys: {Unexpected_Keys[:20]}'
        )

    if Shape_Mismatch_Keys:
        raise RuntimeError(
            f'Checkpoint shape mismatches: {Shape_Mismatch_Keys[:20]}'
        )

    Expected_SMP_Keys = {
        Key for Key in Boundary_State_Dict
        if Key.startswith('Model.')
    }

    Transferred_Keys = set(Transferred_State_Dict)

    Missing_SMP_Keys = Expected_SMP_Keys - Transferred_Keys

    if Missing_SMP_Keys:
        raise RuntimeError(
            f'Missing SMP parameters: {sorted(Missing_SMP_Keys)[:20]}'
        )

    Load_Result = Model.load_state_dict(
        Transferred_State_Dict,
        strict=False
    )

    Unexpected_Missing_Keys = [
        Key for Key in Load_Result.missing_keys
        if not Key.startswith('Boundary_Head.')
    ]

    if Unexpected_Missing_Keys or Load_Result.unexpected_keys:
        raise RuntimeError(
            'Checkpoint loading verification failed. '
            f'Missing keys: {Unexpected_Missing_Keys}; '
            f'Unexpected keys: {Load_Result.unexpected_keys}'
        )

    print(f'Initialized from: {INIT_CHECKPOINT}')
    print(f'Transferred SMP parameters: {len(Transferred_State_Dict)}')
    print(f'New Boundary parameters: {len(Load_Result.missing_keys)}')
    print('SMP checkpoint transfer verified.')

    return Model


# 6. Train One Epoch
def Train_One_Epoch(
    Model,
    Train_Dataloader,
    Segmentation_Criterion,
    Optimizer,
    Device,
    Scaler,
    Use_AMP
):

    Model.train()

    Total_Loss = 0.0
    Total_Segmentation_Loss = 0.0
    Total_Boundary_Loss = 0.0
    Total_Samples = 0

    for Batch_Index, Batch in enumerate(Train_Dataloader, start=1):

        Images = Batch['image'].to(Device, non_blocking=True)

        Masks = Batch['mask'].to(Device, non_blocking=True).long()

        Boundary_Target = Generate_Boundary_Target(Masks)

        Optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda', enabled=Use_AMP):

            Outputs = Model(Images)

            Segmentation_Logits = Outputs['segmentation']
            Boundary_Logits = Outputs['boundary']

            Segmentation_Loss = Segmentation_Criterion(Segmentation_Logits, Masks)

            Boundary_Loss = F.binary_cross_entropy_with_logits(Boundary_Logits, Boundary_Target)

            Loss = (Segmentation_Loss + BOUNDARY_WEIGHT * Boundary_Loss)

        if not torch.isfinite(Loss):
            raise RuntimeError(f'Non-finite loss at batch {Batch_Index}')

        if Use_AMP:
            Scaler.scale(Loss).backward()
            Scaler.step(Optimizer)
            Scaler.update()
        else:
            Loss.backward()
            Optimizer.step()

        Batch_Size = Images.size(0)

        Total_Loss += Loss.detach().item() * Batch_Size

        Total_Segmentation_Loss += (Segmentation_Loss.detach().item() * Batch_Size)

        Total_Boundary_Loss += (Boundary_Loss.detach().item() * Batch_Size)

        Total_Samples += Batch_Size

        if (Batch_Index % 25 == 0 or Batch_Index == len(Train_Dataloader)):

            print(
                f'Batch [{Batch_Index}/{len(Train_Dataloader)}] '
                f'Loss={Loss.detach().item():.6f} '
                f'Seg={Segmentation_Loss.detach().item():.6f} '
                f'Boundary={Boundary_Loss.detach().item():.6f}'
            )

    if Total_Samples == 0:
        raise RuntimeError('No training batches were processed')

    return {
        'loss': Total_Loss / Total_Samples,
        'segmentation_loss': (Total_Segmentation_Loss / Total_Samples),
        'boundary_loss': (Total_Boundary_Loss / Total_Samples)
    }


# 7. Original-Image Validation
def Evaluate_Boundary_Model(Model, Valid_Dataset, Valid_Dataloader, Device):

    Model.eval()

    Probability_Sums = {}
    Probability_Counts = {}

    with torch.inference_mode():

        for Batch in Valid_Dataloader:

            Images = Batch['image'].to(Device, non_blocking=True)

            Image_Paths = Batch['image_path']
            Patch_Xs = Batch['x']
            Patch_Ys = Batch['y']

            Outputs = Model(Images)

            Probabilities = torch.softmax(Outputs['segmentation'], dim=1).float().cpu()

            for Index, Image_Path in enumerate(Image_Paths):

                Image_Path = str(Image_Path)

                X = int(Patch_Xs[Index].item())
                Y = int(Patch_Ys[Index].item())

                Height, Width = Get_Original_Image_Size(Image_Path)

                if Image_Path not in Probability_Sums:

                    Probability_Sums[Image_Path] = torch.zeros((NUM_CLASSES, Height, Width), dtype=torch.float32)

                    Probability_Counts[Image_Path] = torch.zeros((Height, Width), dtype=torch.float32)

                Patch_Probabilities = Probabilities[Index]

                Patch_Height = Patch_Probabilities.shape[1]
                Patch_Width = Patch_Probabilities.shape[2]

                End_Y = min(Y + Patch_Height, Height)
                End_X = min(X + Patch_Width, Width)

                Actual_Height = End_Y - Y
                Actual_Width = End_X - X

                if Actual_Height <= 0 or Actual_Width <= 0:
                    raise ValueError(f'Invalid patch coordinates: {Image_Path}')

                Probability_Sums[Image_Path][
                    :,
                    Y:End_Y,
                    X:End_X
                ] += Patch_Probabilities[
                    :,
                    :Actual_Height,
                    :Actual_Width
                ]

                Probability_Counts[Image_Path][
                    Y:End_Y,
                    X:End_X
                ] += 1

    Original_Predictions = {}

    for Image_Path, Probability_Sum in Probability_Sums.items():

        Count_Map = Probability_Counts[Image_Path]

        if torch.any(Count_Map == 0):
            raise RuntimeError(f'Uncovered pixels detected: {Image_Path}')

        Average_Probabilities = (Probability_Sum / Count_Map.unsqueeze(0))

        Original_Predictions[Image_Path] = torch.argmax(Average_Probabilities, dim=0)

    Expected_Image_Count = len(Valid_Dataset.Base_Dataset.Image_Paths)

    if len(Original_Predictions) != Expected_Image_Count:
        raise RuntimeError('Validation image reconstruction count mismatch')

    Validation_mIoU, Image_Metrics = Evaluate_Original_Images(Valid_Dataset=Valid_Dataset, Original_Predictions=Original_Predictions)

    return float(Validation_mIoU.item()), Image_Metrics


# 8. Save Checkpoint and Log
def Save_Checkpoint(
    Model,
    Optimizer,
    Epoch,
    Train_Results,
    Validation_mIoU,
    Best_mIoU,
    Checkpoint_Path
):

    torch.save(
        {
            'epoch': Epoch,
            'model_state_dict': Model.state_dict(),
            'optimizer_state_dict': Optimizer.state_dict(),
            'train_loss': Train_Results['loss'],
            'segmentation_loss': Train_Results['segmentation_loss'],
            'boundary_loss': Train_Results['boundary_loss'],
            'validation_mIoU': Validation_mIoU,
            'best_mIoU': Best_mIoU,
            'boundary_weight': BOUNDARY_WEIGHT,
            'dice_weight': DICE_WEIGHT,
            'random_seed': RANDOM_SEED
        },
        Checkpoint_Path
    )


def Save_Epoch_Log(
    Log_Path,
    Epoch,
    Train_Results,
    Validation_mIoU,
    Learning_Rate
):

    File_Exists = Log_Path.exists()

    with open(Log_Path, mode='a', newline='', encoding='utf-8') as Log_File:

        Writer = csv.writer(Log_File)

        if not File_Exists:
            Writer.writerow([
                'epoch',
                'train_loss',
                'segmentation_loss',
                'boundary_loss',
                'validation_mIoU',
                'learning_rate'
            ])

        Writer.writerow([
            Epoch,
            f'{Train_Results["loss"]:.6f}',
            f'{Train_Results["segmentation_loss"]:.6f}',
            f'{Train_Results["boundary_loss"]:.6f}',
            f'{Validation_mIoU:.6f}',
            f'{Learning_Rate:.8e}'
        ])


# 9. Full Training Pipeline
def Train_Boundary_Model(
    Epochs=DEFAULT_EPOCHS,
    Learning_Rate=DEFAULT_LEARNING_RATE,
    Use_AMP=True,
    Early_Stopping=True,
    Early_Stopping_Patience=EARLY_STOPPING_PATIENCE,
    Early_Stopping_Min_Delta=EARLY_STOPPING_MIN_DELTA
):

    if Epochs < 1:
        raise ValueError('Epochs must be at least 1')

    if Learning_Rate <= 0:
        raise ValueError('Learning rate must be positive')

    if Early_Stopping_Patience < 1:
        raise ValueError('Early stopping patience must be at least 1')

    if Early_Stopping_Min_Delta < 0:
        raise ValueError('Early stopping min_delta must be non-negative')


    Set_Seed()

    Device = Set_Device()
    Use_AMP = Use_AMP and Device.type == 'cuda'

    EXPERIMENT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    EXPERIMENT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    Best_Checkpoint_Path = (EXPERIMENT_CHECKPOINT_DIR / 'best_model.pth')

    Last_Checkpoint_Path = (EXPERIMENT_CHECKPOINT_DIR / 'last_model.pth')

    Log_Path = (EXPERIMENT_LOG_DIR / 'training_log.csv')

    # 새 실험 폴더에 기존 결과가 있으면 덮어쓰지 않음
    Existing_Artifacts = [
        Path_Path for Path_Path in [
            Best_Checkpoint_Path,
            Last_Checkpoint_Path,
            Log_Path
        ]
        if Path_Path.exists()
    ]

    if Existing_Artifacts:
        raise FileExistsError(
            'Experiment outputs already exist. '
            f'Use a new experiment name or archive existing files: {Existing_Artifacts}'
        )

    Train_Dataset, Train_Dataloader = Get_Train_Dataloader(Device)

    Valid_Dataset, Valid_Dataloader = Get_Valid_Dataloader(Device)

    Model = Get_Boundary_Model().to(Device)

    Model = Load_Initial_Checkpoint(Model=Model, Device=Device)

    Segmentation_Criterion = Get_Loss(Loss_Type='ce_dice', Dice_Weight=DICE_WEIGHT).to(Device)

    Optimizer = torch.optim.AdamW(Model.parameters(), lr=Learning_Rate, weight_decay=WEIGHT_DECAY)

    Scaler = torch.amp.GradScaler('cuda', enabled=Use_AMP)

    Best_mIoU = float('-inf')

    Early_Stopping_Best_mIoU = float('-inf')
    Early_Stopping_Counter = 0

    print('=' * 70)
    print(f'Experiment: {EXPERIMENT_NAME}')
    print(f'Epochs: {Epochs}')
    print(f'Learning Rate: {Learning_Rate:.8e}')
    print(f'Dice Weight: {DICE_WEIGHT}')
    print(f'Boundary Weight: {BOUNDARY_WEIGHT}')
    print(f'Mixed Precision: {Use_AMP}')
    print(f'Checkpoint Directory: {EXPERIMENT_CHECKPOINT_DIR}')
    print('=' * 70)

    for Epoch in range(1, Epochs + 1):

        print(f'\nEpoch [{Epoch}/{Epochs}]')

        Train_Results = Train_One_Epoch(
            Model=Model,
            Train_Dataloader=Train_Dataloader,
            Segmentation_Criterion=Segmentation_Criterion,
            Optimizer=Optimizer,
            Device=Device,
            Scaler=Scaler,
            Use_AMP=Use_AMP
        )

        Validation_mIoU, Image_Metrics = (
            Evaluate_Boundary_Model(
                Model=Model,
                Valid_Dataset=Valid_Dataset,
                Valid_Dataloader=Valid_Dataloader,
                Device=Device
            )
        )

        if not np.isfinite(Validation_mIoU):
            raise RuntimeError(f'Invalid validation mIoU at epoch {Epoch}')

        if Early_Stopping:
            if Validation_mIoU > (Early_Stopping_Best_mIoU + Early_Stopping_Min_Delta):
                Early_Stopping_Best_mIoU = Validation_mIoU
                Early_Stopping_Counter = 0

                print(
                    'Early Stopping: Validation mIoU improved. '
                    'Counter reset.'
                )
            else:
                Early_Stopping_Counter += 1

                print(
                    'Early Stopping Counter: '
                    f'{Early_Stopping_Counter}/'
                    f'{Early_Stopping_Patience}'
                )


        Current_Learning_Rate = (Optimizer.param_groups[0]['lr'])

        print('-' * 70)
        print(f'Train Loss: {Train_Results["loss"]:.6f}')
        print(f'Segmentation Loss: {Train_Results["segmentation_loss"]:.6f}')
        print(f'Boundary Loss: {Train_Results["boundary_loss"]:.6f}')
        print(f'Validation mIoU: {Validation_mIoU:.6f}')
        print('-' * 70)

        Save_Epoch_Log(
            Log_Path=Log_Path,
            Epoch=Epoch,
            Train_Results=Train_Results,
            Validation_mIoU=Validation_mIoU,
            Learning_Rate=Current_Learning_Rate
        )

        Is_Best = Validation_mIoU > Best_mIoU

        if Is_Best:
            Best_mIoU = Validation_mIoU

        Save_Checkpoint(
            Model=Model,
            Optimizer=Optimizer,
            Epoch=Epoch,
            Train_Results=Train_Results,
            Validation_mIoU=Validation_mIoU,
            Best_mIoU=Best_mIoU,
            Checkpoint_Path=Last_Checkpoint_Path
        )

        if Is_Best:
            Save_Checkpoint(
                Model=Model,
                Optimizer=Optimizer,
                Epoch=Epoch,
                Train_Results=Train_Results,
                Validation_mIoU=Validation_mIoU,
                Best_mIoU=Best_mIoU,
                Checkpoint_Path=Best_Checkpoint_Path
            )

            print(f'Best model saved: {Best_Checkpoint_Path}')

        if Device.type == 'cuda':
            torch.cuda.empty_cache()


        if Early_Stopping and Early_Stopping_Counter >= Early_Stopping_Patience:
            print(
                '\nEarly Stopping triggered at '
                f'Epoch {Epoch}/{Epochs}.'
            )
            break


    print('\nBoundary training completed.')
    print(f'Best Validation mIoU: {Best_mIoU:.6f}')
    print(f'Best Checkpoint: {Best_Checkpoint_Path}')
    print(f'Training Log: {Log_Path}')

    return Best_mIoU


# 10. Command Line Entry Point
if __name__ == '__main__':

    Parser = argparse.ArgumentParser(
        description='KoMaP Boundary Multi-Task Training'
    )

    Parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)

    Parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)

    Parser.add_argument('--no-amp', action='store_true')

    Parser.add_argument('--early-stopping', action='store_true', help='Enable early stopping based on validation mIoU')

    Parser.add_argument('--early-stopping-patience', type=int, default=EARLY_STOPPING_PATIENCE, help='Number of consecutive epochs without significant improvement')

    Parser.add_argument('--early-stopping-min-delta', type=float, default=EARLY_STOPPING_MIN_DELTA, help='Minimum mIoU improvement required to reset patience')

    Args = Parser.parse_args()

    Train_Boundary_Model(
        Epochs=Args.epochs,
        Learning_Rate=Args.learning_rate,
        Use_AMP=not Args.no_amp,
        Early_Stopping=Args.early_stopping,
        Early_Stopping_Patience=Args.early_stopping_patience,
        Early_Stopping_Min_Delta=Args.early_stopping_min_delta
    )
