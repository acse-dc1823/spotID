from torch.utils.data import Dataset
from PIL import Image
import os
import random


class LeopardDataset(Dataset):
    def __init__(self, root_dir, transform=None, convert=True):
        """
        Args:
            root_dir (string): Directory with all the leopard images
              (subdirectories per leopard).
            transform (callable, optional): Optional transform to
              be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.leopards = []
        self.images = []
        # Dictionary to map leopard identifiers to integers
        self.label_to_index = {}
        self.convert = convert

        idx = 0  # Start index for mapping
        for leopard_id in os.listdir(root_dir):
            leopard_folder = os.path.join(root_dir, leopard_id)
            if os.path.isdir(leopard_folder):
                if leopard_id not in self.label_to_index:
                    self.label_to_index[leopard_id] = idx
                    idx += 1
                img_files = os.listdir(leopard_folder)
                random.shuffle(img_files)
                for img_file in img_files:
                    self.images.append(os.path.join(leopard_folder, img_file))
                    self.leopards.append(leopard_id)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        if self.convert:
            image = Image.open(img_path).convert('RGB')
        else:
            image = Image.open(img_path)
        label = self.leopards[idx]
        # Convert label from string to integer using the mapping
        label = self.label_to_index[label]

        if self.transform:
            image = self.transform(image)

        return image, label
