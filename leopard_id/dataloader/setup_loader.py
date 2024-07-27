from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import LeopardDataset, LeopardBatchSampler, ResizeTransform

import os
import logging
import torch


class PixelDropout(torch.nn.Module):
    def __init__(self, dropout_prob=0.1, apply_dropout=False):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.apply_dropout = apply_dropout  # Manually controlled

    def forward(self, img):
        if self.apply_dropout:  # Use the manual flag to control dropout application
            dropout_mask = torch.rand_like(img[0, :, :]) < self.dropout_prob
            for c in range(img.shape[0]):
                img[c, :, :][dropout_mask] = 0
        return img


def setup_data_loader(config, project_root):
    root_dir_train = os.path.join(project_root, config["train_data_dir"])
    root_dir_test = os.path.join(project_root, config["test_data_dir"])

    if not os.path.isdir(root_dir_train):
        raise NotADirectoryError(f"Directory {root_dir_train} does not exist.")
    else:
        logging.info(f"Found directory {root_dir_train}")

    def create_transforms(resize_width, resize_height, mean=None, std=None, mean_binary_mask=None, std_binary_mask=None,
                          dropout_prob=0.07, apply_dropout=False):
        common_transforms_list = [
            ResizeTransform(width=resize_width, height=resize_height),
            transforms.ToTensor(),
            PixelDropout(dropout_prob=dropout_prob, apply_dropout=apply_dropout)
        ]

        if mean is not None and std is not None:
            common_transforms_list.append(transforms.Normalize(mean=mean, std=std))
            
        common_transforms = transforms.Compose(common_transforms_list)

        if mean_binary_mask is not None and std_binary_mask is not None:
            binary_transforms_list = [
                ResizeTransform(width=resize_width, height=resize_height),
                transforms.ToTensor(),
                PixelDropout(dropout_prob=dropout_prob, apply_dropout=apply_dropout),
                transforms.Normalize(mean=mean_binary_mask, std=std_binary_mask)
            ]
        
            binary_transforms = transforms.Compose(binary_transforms_list)
        
        else:
            binary_transforms = None

        return common_transforms, binary_transforms

    # Create common transformations
    common_transform, binary_transform = create_transforms(
        resize_width=config["resize_width"],
        resize_height=config["resize_height"],
        mean=config.get("mean_normalize"),
        std=config.get("std_normalize"),
        mean_binary_mask=config.get("mean_normalize_binary_mask"),
        std_binary_mask=config.get("std_normalize_binary_mask"),
        apply_dropout=True
    )

    # There is no gain for a class with just one leopard for cosface
    # whereas for triplet loss it can be used as a negative
    if config["method"] == "cosface":
        skip_singleton_classes = True
    else:
        skip_singleton_classes = False

    train_dataset = LeopardDataset(
        root_dir=root_dir_train,
        transform=common_transform,
        transform_binary=binary_transform,
        mask_dir=config["train_binary_mask_dir"],
        skip_singleton_classes=skip_singleton_classes
    )

    test_dataset = LeopardDataset(
        root_dir=root_dir_test,
        transform=common_transform,
        transform_binary=binary_transform,
        mask_dir=config["test_binary_mask_dir"],
        skip_singleton_classes=skip_singleton_classes
    )

    train_sampler = LeopardBatchSampler(
        train_dataset,
        batch_size=config["batch_size"],
        max_images_indiv=config["max_images_individual_leopard_sampler"],
        verbose=config["verbose"],
    )

    # DataLoader for test should have batch size equal to the number of images
    return DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4), DataLoader(
        test_dataset, batch_size=min(len(test_dataset), 64)
    )
