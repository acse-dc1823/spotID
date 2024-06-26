from torch.utils.data import Dataset, DataLoader, BatchSampler
from PIL import Image
import os
import random


class LeopardDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the leopard images (subdirectories per leopard).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.leopards = []
        self.images = []

        # Iterate through each leopard directory and shuffle images
        for leopard_id in os.listdir(root_dir):
            leopard_folder = os.path.join(root_dir, leopard_id)
            if os.path.isdir(leopard_folder):
                img_files = os.listdir(leopard_folder)
                # Shuffle images to remove any bias (images come sorted through location)
                random.shuffle(img_files)  
                for img_file in img_files:
                    self.images.append(os.path.join(leopard_folder, img_file))
                    self.leopards.append(leopard_id)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.leopards[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
