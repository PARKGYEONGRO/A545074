
# scripts/uncertainty_analysis.py
# EXP-009 예측의 픽셀별 불확실성 분석

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    VALID_IMAGE_DIR,
    CHECKPOINT_DIR,
    CLASS_NAMES,
    NUM_CLASSES,
    DEVICE
)

from src.model import Get_Model
from src.evaluate import (
    Get_Validation_Dataloader,
    Get_Original_Image_Size
)


# 1. Analysis Configuration
EXPERIMENT_NAME = 'exp009'

CHECKPOINT_PATH = (CHECKPOINT_DIR / EXPERIMENT_NAME / 'best_model.pth')

OUTPUT_DIR = (PROJECT_ROOT / 'outputs' / 'analysis' / f'{EXPERIMENT_NAME}_uncertainty')

EPSILON = 1e-8

# 엔트로피가 이 값 이상이면 고불확실 픽셀로 집계
HIGH_UNCERTAINTY_THRESHOLD = 0.5

# 엔트로피 구간별 정확도 분석
UNCERTAINTY_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.000001]


# 2. Device
def Get_Device():

    if DEVICE == 'cuda' and torch.cuda.is_available():
        Analysis_Device = torch.device('cuda')
    else:
        Analysis_Device = torch.device('cpu')

    print(f'Analysis Device: {Analysis_Device}')

    return Analysis_Device


# 3. Load Trained Model
def Load_Trained_Model(Analysis_Device):

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f'Checkpoint not found: {CHECKPOINT_PATH}')

    Model = Get_Model().to(Analysis_Device)

    Checkpoint = torch.load(CHECKPOINT_PATH, map_location=Analysis_Device, weights_only=False)

    if 'model_state_dict' not in Checkpoint:
        raise KeyError('model_state_dict not found in checkpoint')

    Model.load_state_dict(Checkpoint['model_state_dict'])
    Model.eval()

    print(f'Loaded Checkpoint: {CHECKPOINT_PATH}')

    if 'epoch' in Checkpoint:
        print(f'Checkpoint Epoch: {Checkpoint["epoch"]}')

    if 'validation_mIoU' in Checkpoint:
        print(f'Checkpoint Validation mIoU: {Checkpoint["validation_mIoU"]:.6f}')

    return Model


# 4. Reconstruct Original Probability Maps
def Reconstruct_Original_Probabilities(Model, Valid_Dataset, Valid_Dataloader, Analysis_Device):

    # 이미지별 확률 합계와 중첩 횟수
    Probability_Sums = {}
    Probability_Counts = {}

    Model.eval()

    with torch.inference_mode():

        for Batch_Index, Batch in enumerate(Valid_Dataloader, start=1):
            Images = Batch['image'].to(Analysis_Device, non_blocking=True)

            Image_Paths = Batch['image_path']
            Patch_Xs = Batch['x']
            Patch_Ys = Batch['y']

            Logits = Model(Images)

            Patch_Probabilities = torch.softmax(Logits, dim=1).float().cpu()

            for Index, Image_Path in enumerate(Image_Paths):

                Image_Path = str(Image_Path)

                X = int(Patch_Xs[Index].item())
                Y = int(Patch_Ys[Index].item())

                Height, Width = Get_Original_Image_Size(Image_Path)

                if Image_Path not in Probability_Sums:

                    Probability_Sums[Image_Path] = torch.zeros((NUM_CLASSES, Height, Width), dtype=torch.float32)

                    Probability_Counts[Image_Path] = torch.zeros((Height, Width), dtype=torch.float32)

                Patch_Probability = Patch_Probabilities[Index]

                Patch_Height = Patch_Probability.shape[1]
                Patch_Width = Patch_Probability.shape[2]

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
                ] += Patch_Probability[
                    :,
                    :Actual_Height,
                    :Actual_Width
                ]

                Probability_Counts[Image_Path][
                    Y:End_Y,
                    X:End_X
                ] += 1

            if Batch_Index % 25 == 0:
                print(f'Processed Batches: {Batch_Index}/{len(Valid_Dataloader)}')

    # 중첩 패치 확률 평균
    Original_Probabilities = {}

    Expected_Image_Count = len(Valid_Dataset.Base_Dataset.Image_Paths)

    for Image_Path, Probability_Sum in Probability_Sums.items():

        Count_Map = Probability_Counts[Image_Path]

        if torch.any(Count_Map == 0):
            raise RuntimeError(f'Uncovered pixels detected: {Image_Path}')

        Average_Probabilities = (Probability_Sum / Count_Map.unsqueeze(0))

        # 확률 정규화 검증
        Probability_Total = Average_Probabilities.sum(dim=0)

        if not torch.allclose(
            Probability_Total,
            torch.ones_like(Probability_Total),
            atol=1e-4
        ):
            raise RuntimeError(f'Probability normalization failed: {Image_Path}')

        Original_Probabilities[Image_Path] = (Average_Probabilities.numpy())

    if len(Original_Probabilities) != Expected_Image_Count:
        raise RuntimeError(f'Validation image reconstruction count mismatch: Expected={Expected_Image_Count}, Actual={len(Original_Probabilities)}')

    return Original_Probabilities


# 5. Calculate Normalized Predictive Entropy
def Calculate_Uncertainty(Probability_Map):

    Probability_Map = np.clip(Probability_Map, EPSILON, 1.0)

    # 수치 오차를 줄이기 위해 확률 재정규화
    Probability_Map = Probability_Map / np.sum(Probability_Map, axis=0, keepdims=True)

    Entropy_Map = -np.sum(Probability_Map * np.log(Probability_Map), axis=0)

    Normalized_Entropy_Map = (Entropy_Map / np.log(NUM_CLASSES))

    return np.clip(Normalized_Entropy_Map, 0.0, 1.0)


# 6. Save Uncertainty Visualization
def Save_Uncertainty_Visualization(Image_Path, Uncertainty_Map, Prediction_Mask, Output_Image_Dir):

    Image_Path = Path(Image_Path)

    with Image.open(Image_Path) as Image_File:
        Original_Image = np.array(Image_File.convert('RGB'))

    # 원본 해상도 검증
    if Original_Image.shape[:2] != Uncertainty_Map.shape:
        raise ValueError(f'Image and uncertainty size mismatch: {Image_Path}')

    Image_Name = Image_Path.stem.replace('_image', '')

    # 1. 원본 불확실성 지도
    Uncertainty_Path = (Output_Image_Dir / f'{Image_Name}_uncertainty.png')

    plt.figure(figsize=(12, 8))

    plt.imshow(Uncertainty_Map, cmap='inferno', vmin=0.0, vmax=1.0)

    plt.colorbar(label='Normalized Entropy', fraction=0.046, pad=0.04)

    plt.title(f'Uncertainty Map: {Image_Name}')
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(Uncertainty_Path, dpi=200, bbox_inches='tight')

    plt.close()

    # 2. 원본 이미지 위에 불확실성 오버레이
    Overlay_Path = (Output_Image_Dir / f'{Image_Name}_overlay.png')

    plt.figure(figsize=(12, 8))

    plt.imshow(Original_Image)

    Overlay = plt.imshow(Uncertainty_Map, cmap='inferno', vmin=0.0, vmax=1.0, alpha=0.55)

    plt.colorbar(Overlay, label='Normalized Entropy', fraction=0.046, pad=0.04)

    plt.title(f'Uncertainty Overlay: {Image_Name}')
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(Overlay_Path, dpi=200, bbox_inches='tight')

    plt.close()

    # 3. 예측 마스크
    Prediction_Path = (Output_Image_Dir / f'{Image_Name}_prediction.png')

    plt.figure(figsize=(12, 8))

    plt.imshow(Prediction_Mask, vmin=0, vmax=NUM_CLASSES - 1, cmap='tab10', interpolation='nearest')

    plt.title(f'Prediction Mask: {Image_Name}')
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(Prediction_Path, dpi=200, bbox_inches='tight')

    plt.close()

    # 4. 원본 불확실성 값 보존
    np.save(
        Output_Image_Dir / f'{Image_Name}_uncertainty.npy',
        Uncertainty_Map.astype(np.float32)
    )


# 7. Per-Image Statistics
def Calculate_Image_Statistics(Image_Path, Probability_Map, Uncertainty_Map, Valid_Dataset):

    Prediction_Mask = np.argmax(Probability_Map, axis=0).astype(np.uint8)

    Statistics = {
        'image_name': Path(Image_Path).name,
        'height': int(Uncertainty_Map.shape[0]),
        'width': int(Uncertainty_Map.shape[1]),
        'mean_uncertainty': float(
            np.mean(Uncertainty_Map)
        ),
        'median_uncertainty': float(
            np.median(Uncertainty_Map)
        ),
        'p90_uncertainty': float(
            np.percentile(Uncertainty_Map, 90)
        ),
        'high_uncertainty_fraction': float(
            np.mean(
                Uncertainty_Map >= HIGH_UNCERTAINTY_THRESHOLD
            )
        ),
        'pixel_count': int(Uncertainty_Map.size)
    }

    # 검증 정답 마스크와 비교
    Target_Mask = Valid_Dataset.Get_Original_Mask(Image_Path).numpy()

    if Target_Mask.shape != Prediction_Mask.shape:
        raise ValueError(f'Prediction and target size mismatch: {Image_Path}')

    Correct_Map = (
        Prediction_Mask == Target_Mask
    )

    Statistics['pixel_accuracy'] = float(np.mean(Correct_Map))

    Statistics['mean_uncertainty_correct'] = (
        float(np.mean(Uncertainty_Map[Correct_Map]))
        if np.any(Correct_Map)
        else np.nan
    )

    Statistics['mean_uncertainty_wrong'] = (
        float(np.mean(Uncertainty_Map[~Correct_Map]))
        if np.any(~Correct_Map)
        else np.nan
    )

    Statistics['high_uncertainty_error_rate'] = (
        float(
            np.mean(
                ~Correct_Map[
                    Uncertainty_Map >= HIGH_UNCERTAINTY_THRESHOLD
                ]
            )
        )
        if np.any(
            Uncertainty_Map >= HIGH_UNCERTAINTY_THRESHOLD
        )
        else np.nan
    )

    return Statistics, Prediction_Mask, Target_Mask


# 8. Uncertainty Bin Analysis
def Calculate_Uncertainty_Bin_Statistics(Image_Path, Uncertainty_Map, Prediction_Mask, Target_Mask):

    Rows = []

    Correct_Map = Prediction_Mask == Target_Mask

    Bin_Labels = [
        '[0.0, 0.2)',
        '[0.2, 0.4)',
        '[0.4, 0.6)',
        '[0.6, 0.8)',
        '[0.8, 1.0]'
    ]

    for Bin_Index in range(len(UNCERTAINTY_BINS) - 1):

        Lower_Bound = UNCERTAINTY_BINS[Bin_Index]
        Upper_Bound = UNCERTAINTY_BINS[Bin_Index + 1]

        if Bin_Index == len(UNCERTAINTY_BINS) - 2:
            Bin_Mask = (
                (Uncertainty_Map >= Lower_Bound)
                & (Uncertainty_Map <= 1.0)
            )
        else:
            Bin_Mask = (
                (Uncertainty_Map >= Lower_Bound)
                & (Uncertainty_Map < Upper_Bound)
            )

        Pixel_Count = int(np.sum(Bin_Mask))

        Rows.append({
            'image_name': Path(Image_Path).name,
            'uncertainty_bin': Bin_Labels[Bin_Index],
            'pixel_count': Pixel_Count,
            'pixel_fraction': float(
                Pixel_Count / Uncertainty_Map.size
            ),
            'mean_uncertainty': (
                float(np.mean(Uncertainty_Map[Bin_Mask]))
                if Pixel_Count > 0
                else np.nan
            ),
            'pixel_accuracy': (
                float(np.mean(Correct_Map[Bin_Mask]))
                if Pixel_Count > 0
                else np.nan
            ),
            'error_rate': (
                float(np.mean(~Correct_Map[Bin_Mask]))
                if Pixel_Count > 0
                else np.nan
            )
        })

    return Rows


# 9. Main
def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    Image_Output_Dir = OUTPUT_DIR / 'images'

    Image_Output_Dir.mkdir(parents=True, exist_ok=True)

    Analysis_Device = Get_Device()

    Model = Load_Trained_Model(Analysis_Device)

    Valid_Dataset, Valid_Dataloader = (Get_Validation_Dataloader())

    Original_Probabilities = (
        Reconstruct_Original_Probabilities(
            Model=Model,
            Valid_Dataset=Valid_Dataset,
            Valid_Dataloader=Valid_Dataloader,
            Analysis_Device=Analysis_Device
        )
    )

    Image_Statistics_Rows = []
    Bin_Statistics_Rows = []

    print('=' * 70)
    print('EXP-009 Uncertainty Analysis')
    print(f'Validation Images: {len(Original_Probabilities)}')
    print(f'Output Directory: {OUTPUT_DIR}')
    print('=' * 70)

    for Image_Path, Probability_Map in (Original_Probabilities.items()):

        Uncertainty_Map = Calculate_Uncertainty(Probability_Map)

        (
            Statistics,
            Prediction_Mask,
            Target_Mask
        ) = Calculate_Image_Statistics(
            Image_Path=Image_Path,
            Probability_Map=Probability_Map,
            Uncertainty_Map=Uncertainty_Map,
            Valid_Dataset=Valid_Dataset
        )

        Save_Uncertainty_Visualization(
            Image_Path=Image_Path,
            Uncertainty_Map=Uncertainty_Map,
            Prediction_Mask=Prediction_Mask,
            Output_Image_Dir=Image_Output_Dir
        )

        Bin_Rows = Calculate_Uncertainty_Bin_Statistics(
            Image_Path=Image_Path,
            Uncertainty_Map=Uncertainty_Map,
            Prediction_Mask=Prediction_Mask,
            Target_Mask=Target_Mask
        )

        Image_Statistics_Rows.append(Statistics)
        Bin_Statistics_Rows.extend(Bin_Rows)

        print(
            f'{Path(Image_Path).name} | '
            f'mean entropy={Statistics["mean_uncertainty"]:.4f} | '
            f'pixel accuracy={Statistics["pixel_accuracy"]:.4f}'
        )

    # 이미지별 통계
    Image_Statistics_Df = pd.DataFrame(
        Image_Statistics_Rows
    )

    Image_Statistics_Df.to_csv(
        OUTPUT_DIR / 'image_uncertainty_summary.csv',
        index=False,
        encoding='utf-8-sig'
    )

    # 엔트로피 구간별 통계
    Bin_Statistics_Df = pd.DataFrame(Bin_Statistics_Rows)

    Bin_Statistics_Df.to_csv(
        OUTPUT_DIR / 'uncertainty_bin_summary.csv',
        index=False,
        encoding='utf-8-sig'
    )

    print('=' * 70)
    print('Uncertainty Analysis Completed')
    print(f'Summary: {OUTPUT_DIR / "image_uncertainty_summary.csv"}')
    print(f'Bin Summary: {OUTPUT_DIR / "uncertainty_bin_summary.csv"}')
    print(f'Visualizations: {Image_Output_Dir}')
    print('=' * 70)


if __name__ == '__main__':
    main()
