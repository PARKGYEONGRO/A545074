
# scripts/microstructure_analysis.py
# 미세조직 상별 면적 및 연결 영역 정량 분석

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy import ndimage
from scipy.ndimage import find_objects

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
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

OUTPUT_DIR = (PROJECT_ROOT / 'outputs' / 'analysis' / f'{EXPERIMENT_NAME}_microstructure')

# 연결 영역 통계에서만 사용할 최소 영역 크기
# 1이면 모든 연결 영역을 포함
MIN_COMPONENT_AREA = 1



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

    return Model



# 4. Reconstruct Original Prediction Masks
def Reconstruct_Original_Predictions(Model, Valid_Dataset, Valid_Dataloader, Analysis_Device):

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

            Probabilities = torch.softmax(Logits, dim=1).float().cpu()

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

            if Batch_Index % 25 == 0:
                print(f'Processed Batches: {Batch_Index}/{len(Valid_Dataloader)}')

    Original_Predictions = {}

    Expected_Image_Count = len(Valid_Dataset.Base_Dataset.Image_Paths)

    for Image_Path, Probability_Sum in Probability_Sums.items():

        Count_Map = Probability_Counts[Image_Path]

        if torch.any(Count_Map == 0):
            raise RuntimeError(f'Uncovered pixels detected: {Image_Path}')

        Average_Probabilities = (Probability_Sum / Count_Map.unsqueeze(0))

        Prediction_Mask = torch.argmax(Average_Probabilities, dim=0)

        Original_Predictions[Image_Path] = (Prediction_Mask.numpy().astype(np.uint8))

    if len(Original_Predictions) != Expected_Image_Count:
        raise RuntimeError(
            'Validation image reconstruction count mismatch: '
            f'Expected={Expected_Image_Count}, '
            f'Actual={len(Original_Predictions)}'
        )

    return Original_Predictions


# 5. Connected Component Analysis
def Analyze_Connected_Components(Class_Mask):

    # 8방향 연결성을 사용해 연결 영역 식별
    Structure = np.ones((3, 3), dtype=np.uint8)

    Labeled_Mask, Component_Count = ndimage.label(Class_Mask, structure=Structure)

    if Component_Count == 0:
        return []

    Component_Slices = find_objects(Labeled_Mask)

    Component_Areas = np.bincount(
        Labeled_Mask.ravel(),
        minlength=Component_Count + 1
    )[1:]

    Component_Rows = []

    for Component_Index, Component_Slice in enumerate(Component_Slices, start=1):

        if Component_Slice is None:
            continue

        Component_Area = int(Component_Areas[Component_Index - 1])

        if Component_Area < MIN_COMPONENT_AREA:
            continue

        Height = (Component_Slice[0].stop - Component_Slice[0].start)

        Width = (Component_Slice[1].stop - Component_Slice[1].start)

        Bounding_Box_Area = int(Height * Width)

        Equivalent_Diameter = float(np.sqrt(4.0 * Component_Area / np.pi))

        Aspect_Ratio = float(max(Height, Width) / max(min(Height, Width), 1))

        Extent = float(Component_Area / max(Bounding_Box_Area, 1))

        Component_Rows.append({
            'component_id': Component_Index,
            'area_px': Component_Area,
            'bbox_height_px': int(Height),
            'bbox_width_px': int(Width),
            'equivalent_diameter_px': Equivalent_Diameter,
            'aspect_ratio': Aspect_Ratio,
            'extent': Extent
        })

    return Component_Rows


# 6. Analyze One Image Mask
def Analyze_Image_Mask(Image_Path, Mask, Source):

    if Mask.ndim != 2:
        raise ValueError(f'Mask must be 2D: {Image_Path}')

    if not np.isin(Mask, np.arange(NUM_CLASSES)).all():
        raise ValueError(f'Unexpected class ID detected: {Image_Path}')

    Image_Name = Path(Image_Path).name

    Total_Pixels = int(Mask.size)

    Phase_Summary_Rows = []
    Component_Detail_Rows = []

    for Class_ID, Class_Name in enumerate(CLASS_NAMES):

        Class_Mask = (Mask == Class_ID)

        Class_Pixel_Count = int(np.sum(Class_Mask))

        Area_Fraction = (Class_Pixel_Count / Total_Pixels)

        Component_Rows = Analyze_Connected_Components(Class_Mask)

        Component_Areas = [
            Row['area_px']
            for Row in Component_Rows
        ]

        Component_Diameters = [
            Row['equivalent_diameter_px']
            for Row in Component_Rows
        ]

        Component_Aspect_Ratios = [
            Row['aspect_ratio']
            for Row in Component_Rows
        ]

        Component_Extents = [
            Row['extent']
            for Row in Component_Rows
        ]

        Phase_Summary_Rows.append({
            'image_name': Image_Name,
            'source': Source,
            'class_id': Class_ID,
            'class_name': Class_Name,
            'image_height_px': int(Mask.shape[0]),
            'image_width_px': int(Mask.shape[1]),
            'total_pixels': Total_Pixels,
            'class_pixels': Class_Pixel_Count,
            'area_fraction': float(Area_Fraction),
            'area_percentage': float(
                Area_Fraction * 100.0
            ),
            'component_count': len(Component_Rows),
            'component_area_mean_px': (
                float(np.mean(Component_Areas))
                if Component_Areas else np.nan
            ),
            'component_area_median_px': (
                float(np.median(Component_Areas))
                if Component_Areas else np.nan
            ),
            'component_area_p90_px': (
                float(np.percentile(Component_Areas, 90))
                if Component_Areas else np.nan
            ),
            'component_area_max_px': (
                int(np.max(Component_Areas))
                if Component_Areas else 0
            ),
            'equivalent_diameter_mean_px': (
                float(np.mean(Component_Diameters))
                if Component_Diameters else np.nan
            ),
            'aspect_ratio_mean': (
                float(np.mean(Component_Aspect_Ratios))
                if Component_Aspect_Ratios else np.nan
            ),
            'extent_mean': (
                float(np.mean(Component_Extents))
                if Component_Extents else np.nan
            )
        })

        for Component_Row in Component_Rows:

            Component_Detail_Rows.append({
                'image_name': Image_Name,
                'source': Source,
                'class_id': Class_ID,
                'class_name': Class_Name,
                **Component_Row
            })

    return Phase_Summary_Rows, Component_Detail_Rows


# 7. Main
def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    Analysis_Device = Get_Device()

    Model = Load_Trained_Model(Analysis_Device)

    Valid_Dataset, Valid_Dataloader = (Get_Validation_Dataloader())

    Original_Predictions = (
        Reconstruct_Original_Predictions(
            Model=Model,
            Valid_Dataset=Valid_Dataset,
            Valid_Dataloader=Valid_Dataloader,
            Analysis_Device=Analysis_Device
        )
    )

    Phase_Summary_Rows = []
    Component_Detail_Rows = []

    print('=' * 70)
    print('EXP-009 Microstructure Analysis')
    print(f'Validation Images: {len(Original_Predictions)}')
    print(f'Output Directory: {OUTPUT_DIR}')
    print('=' * 70)

    for Image_Path, Prediction_Mask in (Original_Predictions.items()):

        # 1. 예측 마스크 정량 분석
        Pred_Phase_Rows, Pred_Component_Rows = (
            Analyze_Image_Mask(Image_Path=Image_Path, Mask=Prediction_Mask, Source='Prediction')
        )

        Phase_Summary_Rows.extend(Pred_Phase_Rows)

        Component_Detail_Rows.extend(Pred_Component_Rows)

        # 2. 정답 마스크 정량 분석
        Target_Mask = (Valid_Dataset.Get_Original_Mask(Image_Path).numpy()
        )

        if Target_Mask.shape != Prediction_Mask.shape:
            raise ValueError(f'Prediction and target size mismatch: {Image_Path}')

        Target_Phase_Rows, Target_Component_Rows = (
            Analyze_Image_Mask(Image_Path=Image_Path, Mask=Target_Mask, Source='Ground_Truth')
        )

        Phase_Summary_Rows.extend(Target_Phase_Rows)

        Component_Detail_Rows.extend(Target_Component_Rows)

        # 이미지별 면적 분율 합 검증
        Prediction_Area_Sum = sum(
            Row['area_fraction']
            for Row in Pred_Phase_Rows
        )

        Target_Area_Sum = sum(
            Row['area_fraction']
            for Row in Target_Phase_Rows
        )

        if not np.isclose(Prediction_Area_Sum, 1.0, atol=1e-6):
            raise RuntimeError(f'Prediction area fractions do not sum to 1: {Image_Path}')

        if not np.isclose(Target_Area_Sum, 1.0, atol=1e-6):
            raise RuntimeError(f'Ground-truth area fractions do not sum to 1: {Image_Path}')

        print(
            f'{Path(Image_Path).name} | '
            f'Prediction area sum={Prediction_Area_Sum:.6f} | '
            f'Ground-truth area sum={Target_Area_Sum:.6f}'
        )

    # 클래스별 이미지 통계
    Phase_Summary_Df = pd.DataFrame(
        Phase_Summary_Rows
    )

    Phase_Summary_Df.to_csv(
        OUTPUT_DIR / 'phase_area_summary.csv',
        index=False,
        encoding='utf-8-sig'
    )

    # 개별 연결 영역 통계
    Component_Detail_Df = pd.DataFrame(
        Component_Detail_Rows,
        columns=[
            'image_name',
            'source',
            'class_id',
            'class_name',
            'component_id',
            'area_px',
            'bbox_height_px',
            'bbox_width_px',
            'equivalent_diameter_px',
            'aspect_ratio',
            'extent'
        ]
    )

    Component_Detail_Df.to_csv(OUTPUT_DIR / 'component_details.csv', index=False, encoding='utf-8-sig')

    # 데이터셋 전체의 상별 면적 분율 평균
    Dataset_Summary_Df = (
        Phase_Summary_Df
        .groupby(
            ['source', 'class_id', 'class_name'],
            as_index=False
        )
        .agg(
            mean_area_percentage=('area_percentage', 'mean'),
            std_area_percentage=('area_percentage', 'std'),
            mean_component_count=('component_count', 'mean'),
            mean_component_area_px=('component_area_mean_px', 'mean'),
            image_count=('image_name', 'nunique')
        )
    )

    Dataset_Summary_Df.to_csv(OUTPUT_DIR / 'dataset_microstructure_summary.csv', index=False, encoding='utf-8-sig')

    print('=' * 70)
    print('Microstructure Analysis Completed')
    print(f'Phase Summary: {OUTPUT_DIR / "phase_area_summary.csv"}')
    print(f'Component Details: {OUTPUT_DIR / "component_details.csv"}')
    print(f'Dataset Summary: {OUTPUT_DIR / "dataset_microstructure_summary.csv"}')
    print('=' * 70)


if __name__ == '__main__':
    main()
