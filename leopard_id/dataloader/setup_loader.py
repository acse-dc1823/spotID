# author: David Colomer Matachana
# GitHub username: acse-dc1823

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import LeopardDataset, LeopardBatchSampler, ResizeTransform
import os
import logging
import random


class PixelDropout(torch.nn.Module):
    """
    Applies dropout to pixels in the image.

    This transformation randomly sets pixels to zero with probability `dropout_prob`,
    but only if `apply_dropout` is True.
    """

    def __init__(self, dropout_prob=0.1, apply_dropout=False):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.apply_dropout = apply_dropout

    def forward(self, img):
        if self.apply_dropout:
            dropout_mask = torch.rand_like(img[0, :, :]) < self.dropout_prob
            dropout_mask = dropout_mask.unsqueeze(0)  # Add a channel dimension to mask
            dropout_mask = dropout_mask.repeat(img.shape[0], 1, 1)  # Repeat mask for all channels

            # Apply the mask using a copy of the image tensor
            img = img.clone()
            img[dropout_mask] = 0
        return img


class SynchronizedPercentageRandomCrop:
    """
    Applies the same random percentage crop to both an image and its mask.

    This transformer ensures that the image and mask remain aligned after cropping.
    The crop size is determined by a percentage of the original image size per
    side. The method then crops up to this percentage randomly from each side of the image.
    Given the variable nature of leopard camera trap images, this can help the model generalize
    better.

    Attributes:
        max_percent (float): The maximum percentage of the image that can be
            cropped from each side. Should be a value between 0 and 0.5.

    Methods:
        __call__(img, mask=None): Applies the random crop to the image and mask.

    Args:
        img (PIL.Image): The input image to be cropped.
        mask (PIL.Image, optional): The corresponding mask to be cropped.
            If None, only the image will be cropped.

    Returns:
        tuple: A tuple containing:
            - cropped_img (PIL.Image): The cropped input image.
            - cropped_mask (PIL.Image or None): The cropped mask, if a mask was provided.
                Otherwise, None.
    """

    def __init__(self, max_percent):
        if not 0 <= max_percent <= 0.5:
            raise ValueError("max_percent must be between 0 and 0.5")
        self.max_percent = max_percent

    def __call__(self, img, mask):
        width, height = img.size

        # Calculate the maximum pixels to crop
        max_crop_pixels_w = int(width * self.max_percent)
        max_crop_pixels_h = int(height * self.max_percent)

        # Randomly choose how many pixels to crop
        crop_pixels_left = random.randint(0, max_crop_pixels_w)
        crop_pixels_right = random.randint(0, max_crop_pixels_w)
        crop_pixels_top = random.randint(0, max_crop_pixels_h)
        crop_pixels_bottom = random.randint(0, max_crop_pixels_h)

        # Perform the crop
        cropped_img = img.crop(
            (
                crop_pixels_left,
                crop_pixels_top,
                width - crop_pixels_right,
                height - crop_pixels_bottom,
            )
        )
        cropped_mask = (
            mask.crop(
                (
                    crop_pixels_left,
                    crop_pixels_top,
                    width - crop_pixels_right,
                    height - crop_pixels_bottom,
                )
            )
            if mask is not None
            else None
        )

        return cropped_img, cropped_mask


class SynchronizedRotation:
    """
    Applies the same random rotation to both an image and its mask.

    This transformer ensures that the image and mask remain aligned after rotation.
    The rotation angle is randomly chosen within the range [-degrees, degrees].
    This can help improve model generalization for leopard camera trap images, which
    are vary variable in orientation.

    Attributes:
        degrees (float): The maximum rotation angle in degrees. The actual rotation
            will be randomly chosen between -degrees and +degrees.

    Methods:
        __call__(img, mask=None): Applies the random rotation to the image and mask.

    Args:
        img (PIL.Image or torch.Tensor): The input image to be rotated.
        mask (PIL.Image or torch.Tensor, optional): The corresponding mask to be rotated.
            If None, only the image will be rotated.

    Returns:
        tuple: A tuple containing:
            - rotated_img (PIL.Image or torch.Tensor): The rotated input image.
            - rotated_mask (PIL.Image, torch.Tensor, or None): The rotated mask,
              if a mask was provided. Otherwise, None.
    """

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        rotated_img = transforms.functional.rotate(img, angle)
        rotated_mask = transforms.functional.rotate(mask, angle) if mask is not None else None
        return rotated_img, rotated_mask


class CombinedTransform:
    """
    Applies a series of transformations to both an image and its mask.

    This class encapsulates the entire transformation pipeline, including:
    1. Augmentations (rotation, random percentage crop, and color jitter for RGB images)
    2. Resizing
    3. Conversion to tensor
    4. Post-transformations (e.g., normalization and dropout)
    5. Mask-specific transformations

    The order of operations is crucial to ensure that:
    - Augmentations are applied to the PIL image before resizing and conversion to tensor
    - Resizing is applied after augmentations but before conversion to tensor
    - Normalization is applied after conversion to tensor but before dropout
    - Mask transformations are kept separate from image transformations
    """

    def __init__(
        self,
        resize_transform,
        post_transform,
        binary_transform,
        rotation,
        color_jitter,
        random_crop,
        apply_augmentations,
    ):
        self.resize_transform = resize_transform
        self.post_transform = post_transform
        self.binary_transform = binary_transform
        self.rotation = rotation
        self.color_jitter = color_jitter
        self.random_crop = random_crop
        self.apply_augmentations = apply_augmentations

    def __call__(self, img, mask):
        if self.apply_augmentations:
            # Apply rotation
            img, mask = self.rotation(img, mask)

            # Apply random percentage crop
            img, mask = self.random_crop(img, mask)

            # Apply color jitter only to the image
            if self.color_jitter is not None:
                img = self.color_jitter(img)

        # Apply resize transform
        img = self.resize_transform(img)
        if mask is not None:
            mask = self.resize_transform(mask)

        # Convert to tensor
        img = transforms.ToTensor()(img)

        # Apply post-transforms (normalization and dropout)
        img = self.post_transform(img)

        # Apply binary transforms if available
        if self.binary_transform and mask is not None:
            mask = self.binary_transform(mask)
        elif mask is not None:
            mask = transforms.ToTensor()(mask)

        return img, mask


def create_transforms(
    resize_width,
    resize_height,
    mean=None,
    std=None,
    mean_binary_mask=None,
    std_binary_mask=None,
    dropout_prob=0.05,
    apply_dropout=False,
    rotation_degrees=10,
    apply_augmentations=True,
    mask_only=False,
    max_crop_percent=0.07,
):
    """
    Creates a combined transformation pipeline for image and mask processing.

    This function sets up a complex transformation pipeline that includes:
    - Data augmentation (rotation, random percentage crop, and color jitter for RGB images)
    - Resizing
    - Normalization
    - Pixel dropout
    - Mask-specific transformations

    Args:
        resize_width (int): Width to resize images to.
        resize_height (int): Height to resize images to.
        mean (tuple): Mean for normalization.
        std (tuple): Standard deviation for normalization.
        mean_binary_mask (float): Mean for binary mask normalization.
        std_binary_mask (float): Standard deviation for binary mask normalization.
        dropout_prob (float): Probability of dropout.
        apply_dropout (bool): Whether to apply dropout.
        rotation_degrees (int): Maximum degrees of rotation for augmentation.
        apply_augmentations (bool): Whether to apply data augmentation.
        mask_only (bool): Whether to process masks only.
        max_crop_percent (float): Maximum percentage of image to crop (0.1 = 10%).

    Returns:
        CombinedTransform: A transformation object that applies all necessary transformations.
    """
    resize_transform = ResizeTransform(width=resize_width, height=resize_height)

    post_transform_list = []
    if mask_only:
        if mean_binary_mask is not None and std_binary_mask is not None:
            post_transform_list.append(
                transforms.Normalize(mean=mean_binary_mask, std=std_binary_mask)
            )
    else:
        if mean is not None and std is not None:
            post_transform_list.append(transforms.Normalize(mean=mean, std=std))

    post_transform_list.append(PixelDropout(dropout_prob=dropout_prob, apply_dropout=apply_dropout))
    post_transform = transforms.Compose(post_transform_list)

    binary_transform = None
    if not mask_only and mean_binary_mask is not None and std_binary_mask is not None:
        binary_transforms_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_binary_mask, std=std_binary_mask),
            PixelDropout(dropout_prob=dropout_prob, apply_dropout=apply_dropout),
        ]
        binary_transform = transforms.Compose(binary_transforms_list)

    rotation = SynchronizedRotation(degrees=rotation_degrees)
    color_jitter = (
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        if not mask_only
        else None
    )
    random_crop = SynchronizedPercentageRandomCrop(max_percent=max_crop_percent)

    combined_transform = CombinedTransform(
        resize_transform,
        post_transform,
        binary_transform,
        rotation,
        color_jitter,
        random_crop,
        apply_augmentations,
    )

    return combined_transform


def setup_data_loader(config, project_root):
    """
    Sets up data loaders for training and testing. Orchestrator that calls other functions
    in this file.

    This function creates the necessary datasets and data loaders based on the provided
    configuration. It handles the creation of complex transformation pipelines, including
    augmentations, normalization, and mask processing, with support for mask-only mode.

    Args:
        config (dict): Configuration dictionary containing various settings.
        project_root (str): Root directory of the project.

    Returns:
        tuple: A tuple containing the training DataLoader and testing DataLoader.
    """
    mask_only = config.get("mask_only", False)

    if mask_only:
        root_dir_train = os.path.join(project_root, config["train_binary_mask_dir"])
        root_dir_test = os.path.join(project_root, config["test_binary_mask_dir"])
    else:
        root_dir_train = os.path.join(project_root, config["train_data_dir"])
        root_dir_test = os.path.join(project_root, config["test_data_dir"])

    if not os.path.isdir(root_dir_train):
        raise NotADirectoryError(f"Directory {root_dir_train} does not exist.")
    else:
        logging.info(f"Found directory {root_dir_train}")

    if config["method"] == "cosface":
        skip_singleton_classes = True
    else:
        skip_singleton_classes = False

    # Use the new config flag for augmentations
    apply_augmentations = config.get("apply_augmentations", True)

    train_transform = create_transforms(
        resize_width=config["resize_width"],
        resize_height=config["resize_height"],
        mean=config.get("mean_normalize"),
        std=config.get("std_normalize"),
        mean_binary_mask=config.get("mean_normalize_binary_mask"),
        std_binary_mask=config.get("std_normalize_binary_mask"),
        apply_dropout=config.get("apply_dropout_pixels", False),
        apply_augmentations=apply_augmentations,
        mask_only=mask_only,
        max_crop_percent=0.07,  # This will crop up to 10% from each side
    )

    test_transform = create_transforms(
        resize_width=config["resize_width"],
        resize_height=config["resize_height"],
        mean=config.get("mean_normalize"),
        std=config.get("std_normalize"),
        mean_binary_mask=config.get("mean_normalize_binary_mask"),
        std_binary_mask=config.get("std_normalize_binary_mask"),
        apply_dropout=False,
        apply_augmentations=False,  # Never apply augmentations to test data
        mask_only=mask_only,
        max_crop_percent=0,
    )

    train_dataset = LeopardDataset(
        root_dir=root_dir_train,
        transform=train_transform,
        mask_dir=None if mask_only else config["train_binary_mask_dir"],
        skip_singleton_classes=skip_singleton_classes,
        mask_only=mask_only,
    )

    test_dataset = LeopardDataset(
        root_dir=root_dir_test,
        transform=test_transform,
        mask_dir=None if mask_only else config["test_binary_mask_dir"],
        skip_singleton_classes=skip_singleton_classes,
        mask_only=mask_only,
    )

    train_sampler = LeopardBatchSampler(
        train_dataset,
        batch_size=config["batch_size"],
        max_images_indiv=config["max_images_individual_leopard_sampler"],
    )

    return DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4), DataLoader(
        test_dataset, batch_size=min(len(test_dataset), 64)
    )
