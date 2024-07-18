from torch.utils.data import Dataset
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import os
import random


class LeopardDataset(Dataset):
    def __init__(self, root_dir, mask_dir=None, transform=None, transform_binary=None):
        """
        Args:
            root_dir (string): Directory with all the RGB leopard images.
            mask_dir (string, optional): Directory with all the mask images.
            transform (callable, optional): Optional transform to be applied on a sample.
            transform_binary (callable, optional): Optional transform to be applied on a binary mask.
        """
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.transform_binary = transform_binary
        self.images = []
        self.leopards = []
        self.label_to_index = {}

        idx = 0
        for leopard_id in os.listdir(root_dir):
            leopard_folder = os.path.join(root_dir, leopard_id)
            if os.path.isdir(leopard_folder):
                if leopard_id not in self.label_to_index:
                    self.label_to_index[leopard_id] = idx
                    idx += 1
                img_files = os.listdir(leopard_folder)
                random.shuffle(img_files)
                for img_file in img_files:
                    img_path = os.path.join(leopard_folder, img_file)
                    if self.mask_dir:
                        mask_path = os.path.join(self.mask_dir, leopard_id, img_file)
                        if os.path.exists(mask_path):  # Check if the mask exists
                            self.images.append((img_path, mask_path))
                        else:
                            continue  # Skip this image since mask is missing
                    else:
                        self.images.append(img_path)
                    self.leopards.append(leopard_id)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.mask_dir is not None:
            img_path, mask_path = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')  # Load mask as single channel

            if self.transform:
                image = self.transform(image)
            else:
                image = ToTensor()(image)
            
            if self.transform_binary:
                mask = self.transform_binary(mask)
            else:
                mask = ToTensor()(mask)

            # Concatenate image and mask tensors
            combined = torch.cat((image, mask), 0)  # Ensure mask is a single-channel tensor
        else:
            img_path = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            else:
                image = ToTensor()(image)
        
        combined = image
        label = self.leopards[idx]
        label = self.label_to_index[label]

        return combined, label
