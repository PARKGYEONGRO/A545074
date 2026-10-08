import torch
import torch.nn as nn

# 1. Cross Entropy Loss
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.Loss = nn.CrossEntropyLoss()

    # Forward
    def forward(self, Prediction, Target):
        Loss = self.Loss(Prediction, Target)
        return Loss

# 2. Loss Factory
def Get_Loss():
    Loss = CrossEntropyLoss()
    return Loss