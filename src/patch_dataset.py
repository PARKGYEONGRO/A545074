# 실제 학습에 사용할 Patch Dataset 담당

from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from src.config import (
    PATCH_SIZE,
    PATCH_STRIDE
)

from src.dataset import KoMaPDataset


class KoMaPPatchDataset(Dataset):

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

        self.Base_Dataset = KoMaPDataset(
            Image_Dir=self.Image_Dir,
            Mask_Dir=self.Mask_Dir,
            Transform=None,
            Is_Test=self.Is_Test
        )

        self.Patch_Size = PATCH_SIZE
        self.Patch_Stride = PATCH_STRIDE

        self.Patch_Indices = []

        self._Create_Patch_Indices()


    # Patch Coordinate Generation
    def _Create_Patch_Indices(self):

        for Image_Index in range(
            len(self.Base_Dataset)
        ):

            Image_Path = self.Base_Dataset.Image_Paths[
                Image_Index
            ]

            with Image.open(Image_Path) as Image_File:

                Width, Height = Image_File.size

            Y_Positions = self._Get_Patch_Positions(
                Image_Size=Height
            )

            X_Positions = self._Get_Patch_Positions(
                Image_Size=Width
            )

            for Y in Y_Positions:

                for X in X_Positions:

                    self.Patch_Indices.append(
                        (
                            Image_Index,
                            X,
                            Y
                        )
                    )


    # Patch Position Generation
    def _Get_Patch_Positions(
        self,
        Image_Size
    ):

        if Image_Size <= self.Patch_Size:

            return [0]

        Positions = []

        Position = 0

        while True:

            Positions.append(
                Position
            )

            if (
                Position
                + self.Patch_Size
                >= Image_Size
            ):

                break

            Next_Position = (
                Position
                + self.Patch_Stride
            )

            if (
                Next_Position
                + self.Patch_Size
                >= Image_Size
            ):

                Last_Position = (
                    Image_Size
                    - self.Patch_Size
                )

                if Last_Position not in Positions:

                    Positions.append(
                        Last_Position
                    )

                break

            Position = Next_Position

        return Positions

    # Original Mask Retrieval
    def Get_Original_Mask(self, Image_Path):
        Image_Path = Path(Image_Path)

        Mask_Key = Image_Path.stem.replace('_image', '')

        if Mask_Key not in self.Base_Dataset.Mask_Paths:
            raise KeyError(f'Original mask not found: {Mask_Key}')

        Mask_Path = self.Base_Dataset.Mask_Paths[Mask_Key]

        with Image.open(Mask_Path) as Mask_File:
            Mask_Numpy = self.Base_Dataset._Encode_Mask(Mask_File)

        Mask_Tensor = torch.from_numpy(Mask_Numpy.copy()).long()

        return Mask_Tensor

    # Dataset Length
    def __len__(self):

        return len(
            self.Patch_Indices
        )


    # Dataset Item
    def __getitem__(
        self,
        Index
    ):

        Image_Index, X, Y = self.Patch_Indices[
            Index
        ]

        Image_Path = self.Base_Dataset.Image_Paths[
            Image_Index
        ]

        Image_File = Image.open(
            Image_Path
        ).convert('RGB')

        Image_Numpy = np.array(
            Image_File
        )


        # Test
        if self.Is_Test:

            Image_Patch = Image_Numpy[
                Y:Y + self.Patch_Size,
                X:X + self.Patch_Size
            ]

            if self.Transform:

                Transformed = self.Transform(
                    image=Image_Patch
                )

                Image_Patch = Transformed[
                    'image'
                ]

            if isinstance(
                Image_Patch,
                np.ndarray
            ):

                Image_Tensor = torch.from_numpy(
                    Image_Patch
                ).permute(
                    2,
                    0,
                    1
                ).float() / 255.0

            else:

                Image_Tensor = Image_Patch

            return {
                'image': Image_Tensor,
                'image_path': str(Image_Path),
                'x': X,
                'y': Y
            }


        # Train / Valid
        Mask_Path = self.Base_Dataset.Mask_Paths[
            Image_Path.stem.replace(
                '_image',
                ''
            )
        ]

        Mask_File = Image.open(
            Mask_Path
        )

        Mask_Numpy = self.Base_Dataset._Encode_Mask(
            Mask_File
        )

        Image_Patch = Image_Numpy[
            Y:Y + self.Patch_Size,
            X:X + self.Patch_Size
        ]

        Mask_Patch = Mask_Numpy[
            Y:Y + self.Patch_Size,
            X:X + self.Patch_Size
        ]


        # Transform
        if self.Transform:

            Transformed = self.Transform(
                image=Image_Patch,
                mask=Mask_Patch
            )

            Image_Patch = Transformed[
                'image'
            ]

            Mask_Patch = Transformed[
                'mask'
            ]


        # Image Tensor
        if isinstance(
            Image_Patch,
            np.ndarray
        ):

            Image_Tensor = torch.from_numpy(
                Image_Patch
            ).permute(
                2,
                0,
                1
            ).float() / 255.0

        else:

            Image_Tensor = Image_Patch


        # Mask Tensor
        if isinstance(
            Mask_Patch,
            np.ndarray
        ):

            Mask_Tensor = torch.from_numpy(
                Mask_Patch
            ).long()

        else:

            Mask_Tensor = Mask_Patch

        return {
            'image': Image_Tensor,
            'mask': Mask_Tensor,
            'image_path': str(Image_Path),
            'mask_path': str(Mask_Path),
            'x': X,
            'y': Y
        }