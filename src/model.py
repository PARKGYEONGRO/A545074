import torch
import torch.nn as nn
import segmentation_models_pytorch as SMP

from src.config import NUM_CLASSES

# 1. Model Configurtion
ENCODER_NAME = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'


# 2. KoMaP Segementation Model
class KoMaPSegmentationModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.Model = SMP.Unet(
            encoder_name=ENCODER_NAME,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=NUM_CLASSES,
            activation=None
        )

    # Forward
    def forward(self, Image):
        return self.Model(Image)


# 3. Model Factory
def Get_Model():
    Model = KoMaPSegmentationModel()

    return Model