import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

from model import TripletNetwork, cosine_dist
from losses import euclidean_dist
from scripts_preprocessing import crop_images_folder, remove_background_processor, edge_detection

project_root = os.path.dirname(os.path.abspath(__file__))


class InferenceDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None, transform_binary=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform
        self.transform_binary = transform_binary
        self.image_filenames = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_filenames[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.mask_folder:
            mask_name = os.path.join(self.mask_folder, self.image_filenames[idx])
            mask = Image.open(mask_name).convert('L')  # Assuming mask is grayscale
            
            if self.transform:
                image = self.transform(image)
            else:
                image = to_tensor(image)

            if self.transform_binary:
                mask = self.transform_binary(mask)
            else:
                mask = to_tensor(mask)

            # Concatenate image and mask tensors
            combined = torch.cat((image, mask), 0)
        else:
            if self.transform:
                image = self.transform(image)
            else:
                image = to_tensor(image)
            combined = image

        return combined


def load_model(config):
    model = TripletNetwork(backbone_model=config.get("backbone_model"),
                           num_dims=config.get("num_dimensions"),
                           input_channels=config.get("input_channels"))
    model.load_state_dict(torch.load(config["model_path"], map_location='cpu'))
    model.eval()
    return model


def create_transforms(config):
    common_transforms = transforms.Compose([
        transforms.Resize((config['resize_height'], config['resize_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean_normalize'], std=config['std_normalize'])
    ])
    
    # Adjust these transformations if necessary for your masks
    if config.get('mean_normalize_binary_mask') and config.get('std_normalize_binary_mask'):
        binary_transforms = transforms.Compose([
            transforms.Resize((config['resize_height'], config['resize_width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=config['mean_normalize_binary_mask'], std=config['std_normalize_binary_mask'])
        ])
    else:
        binary_transforms = transforms.Compose([
            transforms.Resize((config['resize_height'], config['resize_width'])),
            transforms.ToTensor()
        ])

    return common_transforms, binary_transforms


def run_inference(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)

    base_input_dir = config['unprocessed_image_folder']
    base_crop_output_dir = config['crop_output_folder']
    base_bg_removed_output_dir = config['bg_removed_output_folder']
    base_binary_output_dir = config['base_binary_output_folder']
    if config["preprocess"]:
        crop_images_folder(base_input_dir, base_crop_output_dir,
                           store_full_images=False)
        remove_background_processor(base_crop_output_dir,
                                    base_bg_removed_output_dir)
        edge_detection(base_bg_removed_output_dir, base_binary_output_dir)
    
    common_transform, binary_transform = create_transforms(config)
    dataset = InferenceDataset(
        base_crop_output_dir,
        base_binary_output_dir,
        transform=common_transform,
        transform_binary=binary_transform
    )

    crop_output_folder_absolute = os.path.abspath(config["crop_output_folder"])
    image_filenames_path = os.path.join(config['output_folder'],
                                        'image_filenames.txt')
    with open(image_filenames_path, 'w') as f:
        for filename in dataset.image_filenames:
            full_path = os.path.join(crop_output_folder_absolute, filename)
            f.write(f"{full_path}\n")
    print(f"Image filenames saved to {image_filenames_path}")
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    model = load_model(config)
    embeddings = []

    with torch.no_grad():
        for batch in data_loader:
            output = model(batch)
            embeddings.append(output)

    # Concatenate all collected embeddings along the batch dimension
    embeddings = torch.cat(embeddings, dim=0)

    # Compute distances
    if config.get("distance_metric") == "cosine":
        distance_matrix = cosine_dist(embeddings, embeddings)
    else:  # Default to euclidean if not specified or specified as 'euclidean'
        distance_matrix = euclidean_dist(embeddings, embeddings)

    # Save embeddings and distances
    output_folder = config['output_folder']
    np.save(os.path.join(output_folder, 'embeddings.npy'), embeddings)
    np.save(os.path.join(output_folder, 'distance_matrix.npy'), distance_matrix)

    # Optionally, print or log the information about saved files
    print(f"Embeddings and distance matrix saved in {output_folder}")


if __name__ == "__main__":
    run_inference('config_inference.json')
