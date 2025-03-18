import cv2
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from albumentations.pytorch import ToTensorV2

from albumentations.core.transforms_interface import BasicTransform

def linear_transformation(img):
    s = np.random.rand() + 0.5
    b = np.random.rand() * 100 - 50
    return np.clip(img.astype(np.float32) * s + b, 0, 255)

def add_grayscale_gaussian_noise(image, mean=0, stddev=15):
    # Generate Gaussian noise in grayscale
    noise = np.random.normal(mean, stddev, image.shape[:2])
    if len(image.shape) == 2:
        # Add noise to the grayscale image
        noisy_image = image + noise[:, :, np.newaxis]
    else:
        # Add noise to each channel of the RGB image
        noisy_image = image + noise[:, :, np.newaxis]
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image

def dynamic_resize(image, min_scale=2, max_scale=3, **kwargs):
    height, width = image.shape[:2]
    s = np.random.randint(low=min_scale, high=max_scale+1)
    resized_image = cv2.resize(image, (width // s, height // s))
    resized_image = cv2.resize(resized_image, (width, height))
    return resized_image

def get_dsm_transform():
    # Define the augmentation pipeline
    dsm_transform = A.Compose([
        A.ImageCompression(
            quality_lower=15,
            quality_upper=30,
            p=0.2,
        ),
        A.CoarseDropout(
            always_apply=False, 
            p=0.2, 
            max_holes=10, 
            max_height=100, 
            max_width=100, 
            min_holes=1, 
            min_height=20, 
            min_width=20, 
            fill_value=(0, 0, 0), 
            mask_fill_value=None
        ),
        # A.CoarseDropout(
        #     always_apply=False, 
        #     p=0.2, 
        #     max_holes=10, 
        #     max_height=100, 
        #     max_width=100, 
        #     min_holes=1, 
        #     min_height=20, 
        #     min_width=20, 
        #     fill_value=(255, 255, 255), 
        #     mask_fill_value=None
        # ),
        # A.OneOf([
        A.RandomBrightnessContrast(
            p=0.3, 
            brightness_limit=0.5, 
            contrast_limit=0.2
        ),
            # A.Lambda(
            #     p=0.5,
            #     image=lambda x, **kwargs: linear_transformation(x)
            # )
        # ], p=0.5),
        A.OneOf([
            A.ElasticTransform(
                p=0.5,
                alpha=4, 
                sigma=1, 
                alpha_affine=1
            ),
            A.ElasticTransform(
                p=0.5,
                alpha=14, 
                sigma=4, 
                alpha_affine=1
            ),
            A.ElasticTransform(
                p=0.5,
                alpha=24, 
                sigma=4, 
                alpha_affine=1
            ),
            A.ElasticTransform(
                p=0.5,
                alpha=34, 
                sigma=4, 
                alpha_affine=1
            )
        ], p=0.5),
        A.OneOf([
            A.Blur(
                p=0.5,
                blur_limit= (5, 31)
            ),
            A.MedianBlur(
                p=0.5,
                blur_limit=(3, 5)
            ),
            A.GaussianBlur(
                p=0.5,
                blur_limit=(5, 31)
            ),
            # A.Lambda(image=dynamic_resize, p=0.5),
        ], p=0.3),
        A.Lambda(
            p=0.2,
            image=lambda x, **kwargs: add_grayscale_gaussian_noise(x, mean=0, stddev=15)
        ),
        # A.Normalize(mean=(0.485, ), std=(0.229, )),
        ToTensorV2(),
    ])

    return dsm_transform