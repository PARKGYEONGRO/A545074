# dataset.py / Dataset 클래스 정의

from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from src.config import (
    MASK_COLOR_TO_CLASS,
    NUM_CLASSES
)


class KoMaPDataset(Dataset):

    def __init__(
        self,
        Image_Dir,
        Mask_Dir=None,
        Transform=None,
        Is_Test=False
    ):

        self.Image_Dir = Path(Image_Dir)
        self.Mask_Dir = Path(Mask_Dir) if Mask_Dir else None
        self.Transform = Transform
        self.Is_Test = Is_Test

        self.Image_Paths = sorted(
            self.Image_Dir.glob('*_image.png')
        )

        if len(self.Image_Paths) == 0:

            raise RuntimeError(
                f'이미지를 찾을 수 없습니다: {self.Image_Dir}'
            )

        self.Mask_Paths = {}

        if not self.Is_Test:

            if self.Mask_Dir is None:

                raise ValueError(
                    'Train / Valid Dataset에는 Mask_Dir가 필요합니다'
                )

            Mask_Files = sorted(
                self.Mask_Dir.glob('*_mask.png')
            )

            for Mask_Path in Mask_Files:

                Sample_Key = Mask_Path.stem.replace(
                    '_mask',
                    ''
                )

                self.Mask_Paths[
                    Sample_Key
                ] = Mask_Path

            self._Check_Image_Mask_Matching()


    # Image ↔ Mask Matching
    def _Check_Image_Mask_Matching(self):

        Image_Keys = {
            Image_Path.stem.replace(
                '_image',
                ''
            )
            for Image_Path in self.Image_Paths
        }

        Mask_Keys = set(
            self.Mask_Paths.keys()
        )

        Missing_Masks = Image_Keys - Mask_Keys

        Extra_Masks = Mask_Keys - Image_Keys

        if Missing_Masks:

            raise RuntimeError(
                'Mask가 없는 이미지가 있습니다:\n'
                f'{sorted(Missing_Masks)}'
            )

        if Extra_Masks:

            raise RuntimeError(
                '대응하는 이미지가 없는 Mask가 있습니다:\n'
                f'{sorted(Extra_Masks)}'
            )

    # Mask Encoding
    def _Encode_Mask(self, Mask):

        Mask_RGB = np.array(
            Mask.convert('RGB')
        )

        Height, Width, _ = Mask_RGB.shape

        Encoded_Mask = np.full(
            (Height, Width),
            -1,
            dtype=np.int64
        )

        for Color, Class_ID in MASK_COLOR_TO_CLASS.items():

            Match = np.all(
                Mask_RGB == Color,
                axis=-1
            )

            Encoded_Mask[
                Match
            ] = Class_ID

        Unknown_Pixels = (
            Encoded_Mask == -1
        )

        if np.any(Unknown_Pixels):

            Unknown_Colors = np.unique(
                Mask_RGB[Unknown_Pixels],
                axis=0
            )

            raise ValueError(
                '알 수 없는 Mask 색상이 발견되었습니다:\n'
                f'{Unknown_Colors.tolist()}'
            )

        return Encoded_Mask


    # Dataset Length
    def __len__(self):

        return len(
            self.Image_Paths
        )

    # Dataset Item
    def __getitem__(self, Index):

        Image_Path = self.Image_Paths[
            Index
        ]

        Image_File = Image.open(
            Image_Path
        ).convert('RGB')


        # Test
        if self.Is_Test:

            Image_Numpy = np.array(
                Image_File
            )

            if self.Transform:

                Transformed = self.Transform(
                    image=Image_Numpy
                )

                Image_Numpy = Transformed[
                    'image'
                ]

            if isinstance(
                Image_Numpy,
                np.ndarray
            ):

                Image_Tensor = torch.from_numpy(
                    Image_Numpy
                ).permute(
                    2,
                    0,
                    1
                ).float() / 255.0

            else:

                Image_Tensor = Image_Numpy

            return {
                'image': Image_Tensor,
                'image_path': str(Image_Path)
            }


        # Train / Valid
        Sample_Key = Image_Path.stem.replace(
            '_image',
            ''
        )

        Mask_Path = self.Mask_Paths[
            Sample_Key
        ]

        Mask = Image.open(
            Mask_Path
        )


        # Resolution Check
        if Image_File.size != Mask.size:

            raise ValueError(
                'Image와 Mask의 해상도가 다릅니다.\n'
                f'Image: {Image_Path} {Image_File.size}\n'
                f'Mask : {Mask_Path} {Mask.size}'
            )

        Image_Numpy = np.array(
            Image_File
        )

        Mask_Numpy = self._Encode_Mask(
            Mask
        )


        # Transform
        if self.Transform:

            Transformed = self.Transform(
                image=Image_Numpy,
                mask=Mask_Numpy
            )

            Image_Numpy = Transformed[
                'image'
            ]

            Mask_Numpy = Transformed[
                'mask'
            ]


        # Image Tensor
        if isinstance(
            Image_Numpy,
            np.ndarray
        ):

            Image_Tensor = torch.from_numpy(
                Image_Numpy
            ).permute(
                2,
                0,
                1
            ).float() / 255.0

        else:

            Image_Tensor = Image_Numpy


        # Mask Tensor
        if isinstance(
            Mask_Numpy,
            np.ndarray
        ):

            Mask_Tensor = torch.from_numpy(
                Mask_Numpy
            ).long()

        else:

            Mask_Tensor = Mask_Numpy

        return {
            'image': Image_Tensor,
            'mask': Mask_Tensor,
            'image_path': str(Image_Path),
            'mask_path': str(Mask_Path)
        }