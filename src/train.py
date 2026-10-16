import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    RANDOM_SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    NUM_CLASSES,
    Create_Directories
)

from src.patch_dataset import KoMaPPatchDataset
from src.transforms import Get_Train_Transform
from src.model import Get_Model
from src.losses import Get_Loss
from src.evaluate import Evaluate_Model


# 1. Reproducibility
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


# 2. Work Seed
def Seed_Worker(Worker_ID):

    Worker_Seed = torch.initial_seed() % (2 ** 32)

    np.random.seed(Worker_Seed)
    random.seed(Worker_Seed)


# 3. Training DataLoader
def Get_Train_Dataloader(Device):
    Train_Dataset = KoMaPPatchDataset(
        Image_Dir=TRAIN_IMAGE_DIR,
        Mask_Dir=TRAIN_MASK_DIR,
        Transform=Get_Train_Transform(),
        Is_Test=False
    )

    if len(Train_Dataset) == 0:
        raise RuntimeError('Training Dataset is empty')

    Generator = torch.Generator()
    Generator.manual_seed(RANDOM_SEED)

    Train_Dataloader = DataLoader(
        Train_Dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(Device.type == 'cuda'),
        worker_init_fn=Seed_Worker,
        generator=Generator,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False
    )

    print(f'Train Images: {len(Train_Dataset.Base_Dataset)}')
    print(f'Train Patches: {len(Train_Dataset)}')
    print(f'Train Batches per Epoch: {len(Train_Dataloader)}')

    return Train_Dataset, Train_Dataloader


# 4. Checkpoint Save
def Save_Checkpoint(Model, Optimizer, Epoch, Train_Loss, Validation_mIoU, Best_mIoU, Checkpoint_Path):
    Checkpoint = {
        'epoch': Epoch,
        'model_state_dict': Model.state_dict(),
        'optimizer_state_dict': Optimizer.state_dict(),
        'train_loss': Train_Loss,
        'validation_mIoU': Validation_mIoU,
        'best_mIoU': Best_mIoU,
        'random_seed': RANDOM_SEED
    }

    torch.save(Checkpoint, Checkpoint_Path)


# 5. Csv Logging
def Save_Epoch_Log(Log_Path, Epoch, Train_Loss, Validation_mIoU, Learning_Rate):
    File_Exists = Log_Path.exists()

    with open(Log_Path, mode='a', newline='', encoding='utf-8') as Log_File:
        Writer = csv.writer(Log_File)

        if not File_Exists:
            Writer.writerow(['epoch', 'train_loss', 'validation_mIoU', 'learning_rate'])

        Writer.writerow([Epoch, f'{Train_Loss:.6f}', f'{Validation_mIoU:.6f}', f'{Learning_Rate:.8e}'])


# 6. Train One Epoch
def Train_One_Epoch(Model, Train_Dataloader, Criterion, Optimizer, Device, Scaler, Use_AMP):
    Model.train()

    Total_Loss = 0.0
    Total_Samples = 0

    for Batch_Index, Batch in enumerate(Train_Dataloader, start=1):
        Images = Batch['image'].to(Device, non_blocking=True)

        Masks = Batch['mask'].to(Device, non_blocking=True).long()

        Optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda', enabled=Use_AMP):
            Predictions = Model(Images)

            Loss = Criterion(Predictions, Masks)
            
            if not torch.isfinite(Loss):
                raise RuntimeError(f'Non-finite loss detected at batch {Batch_Index}: {Loss.item()}')

            if Use_AMP:
                Scaler.scale(Loss).backward()
                Scaler.step(Optimizer)
                Scaler.update()

            else:
                Loss.backward()
                Optimizer.step()

            Batch_Size = Images.size(0)

            Total_Loss += Loss.detach().item() * Batch_Size
            Total_Samples += Batch_Size

            if Batch_Index % 25 == 0 or Batch_Index == len(Train_Dataloader):
                print(
                    f'Batch [{Batch_Index}/{len(Train_Dataloader)}]'
                    f'Loss: {Loss.detach().item():.6f}'
                )

    if Total_Samples == 0:
        raise RuntimeError('No training batches were processed')

    Average_Loss = Total_Loss / Total_Samples

    return Average_Loss


# 7. Full Training Pipeline
def Train_Model(
        Epochs=NUM_EPOCHS,
        Use_AMP=True,
        Resume=False,
        Experiment='exp005',
        Init_Checkpoint=None,
        Learning_Rate=LEARNING_RATE,
        Class_Weights=None,
        Loss_Type='ce',
        Dice_Weight=1.0,
        Early_Stopping=False,
        Early_Stopping_Patience=5,
        Early_Stopping_Min_Delta=0.0001
    ):

    if Epochs < 1:
        raise ValueError('Epochs must be at least 1')

    Create_Directories()
    Set_Seed()

    Device = Set_Device()
    Use_AMP = (Use_AMP and Device.type == 'cuda')

    print(f'Mixed Precision: {Use_AMP}')

    Train_Dataset, Train_Dataloader = Get_Train_Dataloader(Device=Device)

    Model = Get_Model().to(Device)

    CLASS_WEIGHTS = Class_Weights

    Criterion = Get_Loss(
        Class_Weights=CLASS_WEIGHTS,
        Loss_Type=Loss_Type,
        Dice_Weight=Dice_Weight
    ).to(Device)

    Optimizer = torch.optim.AdamW(Model.parameters(), lr=Learning_Rate, weight_decay=WEIGHT_DECAY)

    Scaler = torch.amp.GradScaler('cuda', enabled=Use_AMP)

    # Experiment-Specific Paths 
    Experiment_Checkpoint_Dir = CHECKPOINT_DIR / Experiment
    Experiment_Log_Dir = LOG_DIR / Experiment

    Experiment_Checkpoint_Dir.mkdir(parents=True, exist_ok=True)
    Experiment_Log_Dir.mkdir(parents=True, exist_ok=True)

    Best_Checkpoint_Path = Experiment_Checkpoint_Dir / 'best_model.pth'
    Last_Checkpoint_Path = Experiment_Checkpoint_Dir / 'last_model.pth'
    Log_Path = Experiment_Log_Dir / 'training_log.csv'


    # Resume Training
    Start_Epoch = 1
    Best_mIoU = float('-inf')

    Early_Stopping_Best_mIoU = float('-inf')
    Early_Stopping_Counter = 0

    if Early_Stopping_Patience < 1:
        raise ValueError('Early stopping patience must be at least 1')
    if Early_Stopping_Min_Delta < 0:
        raise ValueError('Early stopping min_delta must be non-negative')

    if Init_Checkpoint is not None and not Resume:
        Init_Checkpoint_Path = Path(Init_Checkpoint)

        if not Init_Checkpoint_Path.exists():
            raise FileNotFoundError(f'Initial checkpoint not found: {Init_Checkpoint_Path}')

        Checkpoint = torch.load(Init_Checkpoint_Path, map_location=Device, weights_only=False)

        Model.load_state_dict(Checkpoint['model_state_dict'])

        print(f'Initial Model: {Init_Checkpoint_Path}')
        print('Model weights loaded; optimizer initialized from scratch')

    if Resume:
        if not Last_Checkpoint_Path.exists():
            raise FileNotFoundError(f'Checkpoint not Found: {Last_Checkpoint_Path}')

        Checkpoint = torch.load(Last_Checkpoint_Path, map_location=Device, weights_only=False)

        Model.load_state_dict(Checkpoint['model_state_dict'])
        Optimizer.load_state_dict(Checkpoint['optimizer_state_dict'])

        Start_Epoch = Checkpoint['epoch'] + 1
        Best_mIoU = Checkpoint.get('best_mIoU', float('-inf'))

        print(f'Resuming from epoch: {Checkpoint["epoch"]}')
        print(f'Next epoch: {Start_Epoch}')
        print(f'Previous best mIoU: {Best_mIoU:.6f}')

        if Start_Epoch > Epochs:
            raise ValueError(f'Checkpoint epoch ({Checkpoint["epoch"]})is already at or beyond target epoch ({Epochs})')


    print('=' * 70)
    print('KoMaP Segmentation Training')
    print(f'Experiment: {Experiment}')
    print(f'Epochs: {Epochs}')
    print(f'Batch Size: {BATCH_SIZE}')
    print(f'Learning Rate: {Learning_Rate}')
    print(f'Weight Decay: {WEIGHT_DECAY}')
    print(f'Class Weights: {CLASS_WEIGHTS}')
    print(f'Train Dataset Length: {len(Train_Dataset)}')
    print(f'Checkpoint Directory: {Experiment_Checkpoint_Dir}')
    print(f'Loss Type: {Loss_Type}')
    print(f'Dice Weight: {Dice_Weight}')
    print(f'Early Stopping: {Early_Stopping}')
    print(f'Early Stopping Patience: {Early_Stopping_Patience}')
    print(f'Early Stopping Min Delta: {Early_Stopping_Min_Delta}')
    print(f'Log Path: {Log_Path}')
    print('=' * 70)

    for Epoch in range(Start_Epoch, Epochs +1):
        print(f'\nEpoch [{Epoch}/{Epochs}]')
        print('-' * 70)

        # 7-1. Train
        Train_Loss = Train_One_Epoch(
            Model=Model,
            Train_Dataloader=Train_Dataloader,
            Criterion=Criterion,
            Optimizer=Optimizer,
            Device=Device,
            Scaler=Scaler,
            Use_AMP=Use_AMP
        )

        # 7-2. Original Image Validation
        Validation_Results = Evaluate_Model(Model=Model, Device=Device)

        Validation_mIoU = float(Validation_Results['Validation_mIoU'].item())

        if not np.isfinite(Validation_mIoU):
            raise RuntimeError(
                f'Invalid Validation mIoU at epoch: {Epoch}'
                f'{Validation_mIoU}'
            )

        Current_Learning_Rate = Optimizer.param_groups[0]['lr']

        print('=' * 70)
        print(f'Epoch: {Epoch}/{Epochs}')
        print(f'Train Loss: {Train_Loss:.6f}')
        print(f'Validation mIoU: {Validation_mIoU:.6f}')
        print(f'Learning Rate: {Current_Learning_Rate:.8e}')
        print('='* 70)

        # 7-2-1. Early Stopping Check
        if Early_Stopping:
            if (
                Validation_mIoU
                > Early_Stopping_Best_mIoU + Early_Stopping_Min_Delta
            ):
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

        # 7-3. Save Log
        Save_Epoch_Log(
            Log_Path=Log_Path,
            Epoch=Epoch,
            Train_Loss=Train_Loss,
            Validation_mIoU=Validation_mIoU,
            Learning_Rate=Current_Learning_Rate
        )

        # 7-4. Save Latest Checkpoint
        Is_Best = Validation_mIoU > Best_mIoU
        
        if Is_Best:
            Best_mIoU = Validation_mIoU

        Save_Checkpoint(
            Model=Model,
            Optimizer=Optimizer,
            Epoch=Epoch,
            Train_Loss=Train_Loss,
            Validation_mIoU=Validation_mIoU,
            Best_mIoU=Best_mIoU,
            Checkpoint_Path=Last_Checkpoint_Path
        )

        # 7-5. Save Best Checkpoint
        if Is_Best:
            Save_Checkpoint(
                Model=Model,
                Optimizer=Optimizer,
                Epoch=Epoch,
                Train_Loss=Train_Loss,
                Validation_mIoU=Validation_mIoU,
                Best_mIoU=Best_mIoU,
                Checkpoint_Path=Best_Checkpoint_Path
            )

            print(f'Best Model Saved: {Best_Checkpoint_Path}')

        if Device.type == 'cuda':
            torch.cuda.empty_cache()

        if Early_Stopping and Early_Stopping_Counter >= Early_Stopping_Patience:
            print(
                '\nEarly Stopping triggered at '
                f'Epoch {Epoch}/{Epochs}.'
            )
            print(
                'Best Validation mIoU: '
                f'{Best_mIoU:.6f}'
            )
            break

    print('\nTraining Completed')
    print(f'Best Validation mIoU: {Best_mIoU:.6f}')
    print(f'Best Checkpoint: {Best_Checkpoint_Path}')
    print(f'Last Checkpoint: {Last_Checkpoint_Path}')
    print(f'Training Log: {Log_Path}')

    return Best_mIoU


# 8. Command Line Entry Point
if __name__ == '__main__':
    Parser = argparse.ArgumentParser(description='KoMaP Segmentation Training')

    Parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of training epochs')

    Parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision training')

    Parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint')

    Parser.add_argument('--experiment', type=str, default='exp005', help='Experiment name used for separate checkpoint and log paths')

    Parser.add_argument('--init-checkpoint', type=str, default=None, help='Load model weights from a checkpoint without restoring optimizer')

    Parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='Learning rate for the optimizer')

    Parser.add_argument('--class-weights', type=float, nargs=NUM_CLASSES, default=None, help='Optional class weights: Al Al3Ni Primary_Si Eutectic_Si')

    Parser.add_argument('--loss-type', type=str, choices=['ce', 'ce_dice'], default='ce', help='Loss function type')

    Parser.add_argument('--dice-weight', type=float, default=1.0, help='Weight applied to Dice Loss')
    
    Parser.add_argument('--early-stopping', action='store_true', help='Enable early stopping based on validation mIoU')

    Parser.add_argument('--early-stopping-patience', type=int, default=5, help='Number of consecutive epochs without significant improvement')

    Parser.add_argument('--early-stopping-min-delta', type=float, default=0.0001, help='Minimum mIoU improvement required to reset patience')

    Args = Parser.parse_args()

    if Args.resume and Args.init_checkpoint is not None:
        Parser.error('--resume and --init-checkpoint cannot be used together')

    if Args.learning_rate <= 0:
        Parser.error('--learning-rate must be greater than 0')

    Train_Model(
        Epochs=Args.epochs,
        Use_AMP=not Args.no_amp,
        Resume=Args.resume,
        Experiment=Args.experiment,
        Init_Checkpoint=Args.init_checkpoint,
        Learning_Rate=Args.learning_rate,
        Class_Weights=Args.class_weights,
        Loss_Type=Args.loss_type,
        Dice_Weight=Args.dice_weight,
        Early_Stopping=Args.early_stopping,
        Early_Stopping_Patience=Args.early_stopping_patience,
        Early_Stopping_Min_Delta=Args.early_stopping_min_delta
    )

