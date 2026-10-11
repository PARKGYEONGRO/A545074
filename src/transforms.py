import albumentations as A

# 1. Train Transform
def Get_Train_Transform():
    return A.Compose([
        A.HorizontalFlip(
            p=0.5
        ),

        A.VerticalFlip(
            p=0.5
        ),

        A.RandomRotate90(
            p=0.5
        ),

        # A.Affine(
        #     scale=(0.95, 1.05),
        #     translate_percent=(-0.03, 0.03),
        #     rotate=(-10, 10),
        #     shear=(-3, 3),
        #     interpolation=1,
        #     mask_interpolation=0,
        #     border_mode=0,
        #     fill=0,
        #     fill_mask=0,
        #     p=0.3
        # ),

        A.RandomBrightnessContrast(
            brightness_limit=0.10,
            contrast_limit=0.10,
            p=0.3
        )
    ])

# 2. Valid Transform
def Get_Valid_Transform():
    return A.Compose([])

# 3. Test Transform
def Get_Test_Transform():
    return A.Compose([])