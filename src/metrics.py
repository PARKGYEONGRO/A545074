import torch

# 1. IoU Calculation
def Calculate_IoU(Prediction, Target, Num_Classes):
    Prediction = torch.argmax(Prediction, dim=1)

    IoU_List = []

    for Class_ID in range(Num_Classes):
        Prediction_Class = Prediction == Class_ID
        Target_Class = Target == Class_ID

        Intersection = torch.logical_and(Prediction_Class, Target_Class).sum().float()

        Union = torch.logical_or(Prediction_Class, Target_Class).sum().float()

        if Union == 0:
            IoU = torch.tensor(float('nan'), device=Prediction.device)
        else:
            IoU = Intersection / Union

        IoU_List.append(IoU)

    return torch.stack(IoU_List)


# 2. Mean IoU Calculation
def Calculate_mIoU(Prediction, Target, Num_Classes):
    IoU_List = Calculate_IoU(Prediction, Target, Num_Classes)

    Valid_IoU = IoU_List[
        ~torch.isnan(IoU_List)
    ]

    if Valid_IoU.numel() == 0:
        return torch.tensor(0.0, device=Prediction.device)

    mIoU = Valid_IoU.mean()

    return mIoU


# 3. Metric Calculation
def Calculate_Metrics(Prediction, Target, Num_Classes):
    IoU_List = Calculate_IoU(Prediction, Target, Num_Classes)

    mIoU = Calculate_mIoU(Prediction, Target, Num_Classes)

    return {
        'IoU': IoU_List,
        'mIoU': mIoU
    }


# 4. Original Image IoU Calculation
def Calculate_IoU_From_Masks(Prediction_Mask, Target_Mask, Num_Classes):
    if Prediction_Mask.shape != Target_Mask.shape:
        raise ValueError('Prediction Mask와 Target Mask의 Shape이 다릅니다.')

    if Prediction_Mask.ndim != 2:
        raise ValueError('원본 이미지 Mask는 2차원이어야 합니다.')

    Prediction_Mask = Prediction_Mask.long()
    Target_Mask = Target_Mask.long()

    IoU_List = []

    for Class_ID in range(Num_Classes):
        Prediction_Class = (Prediction_Mask == Class_ID)

        Target_Class = (Target_Mask == Class_ID)

        Intersection = torch.logical_and(Prediction_Class, Target_Class).sum().float()

        Union = torch.logical_or(Prediction_Class, Target_Class).sum().float()

        if Union.item() == 0:
            IoU = torch.tensor(float('nan'), device=Prediction_Mask.device)
        else:
            IoU = Intersection / Union

        IoU_List.append(IoU)

    return torch.stack(IoU_List)


# 5. Original Image Metric Calculation
def Calculate_Image_Metrics(Prediction_Mask, Target_Mask, Num_Classes):
    IoU_List = Calculate_IoU_From_Masks(
        Prediction_Mask=Prediction_Mask,
        Target_Mask=Target_Mask,
        Num_Classes=Num_Classes
    )

    Valid_IoU = IoU_List[
        ~torch.isnan(IoU_List)
    ]

    if Valid_IoU.numel() == 0:
        Image_mIoU = torch.tensor(0.0, device=Prediction_Mask.device)

    else:
        Image_mIoU = Valid_IoU.mean()

    return {
        'IoU': IoU_List,
        'mIoU': Image_mIoU
    }


# 6. Validation mIoU Calculation
def Calculate_Validation_mIoU(Image_mIoU_List):
    if len(Image_mIoU_List) == 0:
        raise ValueError('Validation 평가 결과가 없습니다.')

    Image_mIoU_Tensor = torch.stack([
        torch.as_tensor(
            Image_mIoU,
            dtype=torch.float32,
        ).reshape(())
        for Image_mIoU in Image_mIoU_List
    ])

    Valid_Image_mIoU = Image_mIoU_Tensor[
        ~torch.isnan(Image_mIoU_Tensor)
    ]

    if Valid_Image_mIoU.numel() == 0:
        return torch.tensor(0.0)

    Validation_mIoU = Valid_Image_mIoU.mean()

    return Validation_mIoU