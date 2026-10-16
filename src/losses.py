import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import NUM_CLASSES


# 1. Cross Entropy Loss
class CrossEntropyLoss(nn.Module):
    def __init__(self, Class_Weights=None):
        super().__init__()

        if Class_Weights is not None:
            Class_Weights = torch.tensor(Class_Weights, dtype=torch.float32)

        self.Loss = nn.CrossEntropyLoss(weight=Class_Weights)

    # Forward
    def forward(self, Prediction, Target):
        return self.Loss(Prediction, Target)
 

# 2. Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, Num_Classes=NUM_CLASSES, Smooth=1.0):
        super().__init__()

        self.Num_Classes = Num_Classes
        self.Smooth = Smooth

    def forward(self, Prediction, Target):
        Probabilities = torch.softmax(Prediction, dim=1)

        Target_One_Hot = F.one_hot(Target.long(), num_classes=self.Num_Classes).permute(0, 3, 1, 2).to(Probabilities.dtype)

        Dimensions = (0, 2, 3)

        Intersection = (Probabilities * Target_One_Hot).sum(dim=Dimensions)

        Denominator = (
            Probabilities.sum(dim=Dimensions)
            + Target_One_Hot.sum(dim=Dimensions)
        )

        Dice_Score = (2.0 * Intersection + self.Smooth) / (Denominator + self.Smooth)

        Loss = 1.0 - Dice_Score.mean()

        return Loss


# 3. Combined Cross Entropy + Dice Loss
class CombinedLoss(nn.Module):
    def __init__(self, Class_Weights=None, Dice_Weight=1.0,):
        super().__init__()

        if Dice_Weight < 0:
            raise ValueError('Dice_Weight must be non-negative')

        self.Cross_Entropy = CrossEntropyLoss(Class_Weights=Class_Weights)

        self.Dice = DiceLoss(Num_Classes=NUM_CLASSES)

        self.Dice_Weight = Dice_Weight

    def forward(self, Prediction, Target):
        CE_Loss = self.Cross_Entropy(Prediction, Target)
        Dice_Loss = self.Dice(Prediction, Target)

        return CE_Loss + self.Dice_Weight * Dice_Loss


# 4. Loss Factory
def Get_Loss(Class_Weights=None, Loss_Type='ce', Dice_Weight=1.0):
    if Class_Weights is not None and len(Class_Weights) != NUM_CLASSES:
        raise ValueError(f'Expected {NUM_CLASSES} class weights, but received {len(Class_Weights)}')

    if Loss_Type == 'ce':
        return CrossEntropyLoss(Class_Weights=Class_Weights)
    elif Loss_Type == 'ce_dice':
        return CombinedLoss(Class_Weights=Class_Weights, Dice_Weight=Dice_Weight)

    raise ValueError(f'Unsupported Loss_Type: {Loss_Type}')
