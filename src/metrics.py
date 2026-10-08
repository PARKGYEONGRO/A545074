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