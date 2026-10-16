# src/boundary_model.py
# 경계 인식 다중 과제 분할 모델

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as SMP

from src.config import NUM_CLASSES
from src.model import ENCODER_NAME

class KoMaPBoundaryMultiTaskModel(nn.Module):

    def __init__(self):
        super().__init__()

        # 기존 U-Net 구조 유지
        self.Model = SMP.Unet(
            encoder_name=ENCODER_NAME,
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
            activation=None
        )

        # Decoder의 마지막 출력 채널
        Decoder_Channels = self.Model.segmentation_head[0].in_channels

        # 경계 예측 헤드
        self.Boundary_Head = nn.Sequential(
            nn.Conv2d(
                Decoder_Channels,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, Image):
        Encoder_Features = self.Model.encoder(Image)

        Decoder_Features = self.Model.decoder(Encoder_Features)

        Segmentation_Logits = self.Model.segmentation_head(Decoder_Features)

        Boundary_Logits = self.Boundary_Head(Decoder_Features)

        #경계 출력을 원본 패치 크기로 복원
        Boundary_Logits = F.interpolate(
            Boundary_Logits,
            size=Image.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        return {
            'segmentation': Segmentation_Logits,
            'boundary': Boundary_Logits
        }


def Get_Boundary_Model():
    return KoMaPBoundaryMultiTaskModel()