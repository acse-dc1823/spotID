# author: David Colomer Matachana
# GitHub username: acse-dc1823

from torch.utils.data import Dataset
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import os
import random


class LeopardDataset(Dataset):
    def __init__(
        self, root_dir, mask_dir=None, transform=None, skip_singleton_classes=False, mask_only=False
    ):
        """
        Args:
            root_dir (string): Directory with all the RGB leopard images or mask images if
                               mask_only is True.
            mask_dir (string, optional): Directory with all the mask images. Not used if
                                         mask_only is True.
            transform (callable, optional): Optional transform to be applied on a sample.
            skip_singleton_classes (bool): Whether to skip classes with only one image.
            mask_only (bool): If True, only use the mask channel (1 channel input).
        """
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_only = mask_only
        self.images = []
        self.leopards = []
        self.label_to_index = {}
        self.skip_singleton_classes = skip_singleton_classes

        if mask_only and mask_dir:
            raise ValueError(
                "When mask_only is True, use root_dir for mask images and do not provide mask_dir."
            )

        idx = 0
        for leopard_id in os.listdir(root_dir):
            leopard_folder = os.path.join(root_dir, leopard_id)
            if os.path.isdir(leopard_folder):
                img_files = os.listdir(leopard_folder)
                if not self.skip_singleton_classes or (
                    self.skip_singleton_classes and len(img_files) > 1
                ):
                    if leopard_id not in self.label_to_index:
                        self.label_to_index[leopard_id] = idx
                        idx += 1
                    random.shuffle(img_files)
                    for img_file in img_files:
                        if mask_only:
                            mask_path = os.path.join(leopard_folder, img_file)
                            self.images.append(mask_path)
                        elif mask_dir:
                            img_path = os.path.join(leopard_folder, img_file)
                            mask_path = os.path.join(mask_dir, leopard_id, img_file)
                            if os.path.exists(mask_path):
                                self.images.append((img_path, mask_path))
                            else:
                                continue  # Skip this image since mask is missing
                        else:
                            img_path = os.path.join(leopard_folder, img_file)
                            self.images.append(img_path)
                        self.leopards.append(leopard_id)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.mask_only:
            mask_path = self.images[idx]
            mask = Image.open(mask_path).convert("L")  # Load mask as single channel
            if self.transform:
                mask, _ = self.transform(mask, None)
            else:
                mask = ToTensor()(mask)
            combined = mask
        elif self.mask_dir:
            img_path, mask_path = self.images[idx]
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")  # Load mask as single channel

            if self.transform:
                image, mask = self.transform(image, mask)
            else:
                image = ToTensor()(image)
                mask = ToTensor()(mask)

            # Concatenate image and mask tensors
            combined = torch.cat((image, mask), 0)
        else:
            img_path = self.images[idx]
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image, _ = self.transform(image, None)
            else:
                image = ToTensor()(image)
            combined = image

        label = self.leopards[idx]
        label = self.label_to_index[label]
        return combined, label
