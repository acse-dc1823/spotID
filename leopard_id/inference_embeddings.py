import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

from model import EmbeddingNetwork
from losses import euclidean_dist, cosine_dist
from scripts_preprocessing import crop_images_folder, remove_background_processor, edge_detection

project_root = os.path.dirname(os.path.abspath(__file__))

class InferenceDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None, transform_binary=None, existing_files=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform
        self.transform_binary = transform_binary
        self.pairs = []
        
        existing_files = set(existing_files) if existing_files else set()
        
        for dirpath, _, filenames in os.walk(image_folder):
            for filename in filenames:
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.abspath(os.path.join(dirpath, filename))

                    if img_path not in existing_files:
                        if self.mask_folder:
                            relative_path = os.path.relpath(img_path, image_folder)
                            mask_path = os.path.join(mask_folder, relative_path)
                            if os.path.exists(mask_path):
                                self.pairs.append((img_path, mask_path))
                        else:
                            self.pairs.append((img_path, None))
        
        # Sort pairs to ensure consistent ordering
        self.pairs.sort(key=lambda x: x[0])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]
        image = Image.open(img_name).convert('RGB')
        
        if mask_name:
            mask = Image.open(mask_name).convert('L')
            
            if self.transform:
                image = self.transform(image)
            else:
                image = to_tensor(image)

            if self.transform_binary:
                mask = self.transform_binary(mask)
            else:
                mask = to_tensor(mask)

            combined = torch.cat((image, mask), 0)
        else:
            if self.transform:
                image = self.transform(image)
            else:
                image = to_tensor(image)
            combined = image

        return combined, img_name, mask_name

def load_model(config):
    model = EmbeddingNetwork(backbone_model=config.get("backbone_model"),
                           num_dims=config.get("num_dimensions"),
                           input_channels=config.get("input_channels"))
    model_path = os.path.abspath(os.path.join(project_root, config["model_path"]))

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def create_transforms(config):
    common_transforms = transforms.Compose([
        transforms.Resize((config['resize_height'], config['resize_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean_normalize'], std=config['std_normalize'])
    ])
    
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
    config_path = os.path.abspath(config_path)
    
    with open(config_path, 'r') as file:
        config = json.load(file)

    # Convert relative paths to absolute paths based on script location
    output_folder = os.path.abspath(os.path.join(project_root, config['output_folder']))
    base_input_dir = os.path.abspath(os.path.join(project_root, config['unprocessed_image_folder']))
    base_crop_output_dir = os.path.abspath(os.path.join(project_root, config['crop_output_folder']))
    base_bg_removed_output_dir = os.path.abspath(os.path.join(project_root, config['bg_removed_output_folder']))
    base_binary_output_dir = os.path.abspath(os.path.join(project_root, config['base_binary_output_folder']))

    # Create output folders if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(base_crop_output_dir, exist_ok=True)
    os.makedirs(base_bg_removed_output_dir, exist_ok=True)
    os.makedirs(base_binary_output_dir, exist_ok=True)

    if config["preprocess"]:
        crop_images_folder(base_input_dir, base_crop_output_dir, store_full_images=False)
        remove_background_processor(base_crop_output_dir, base_bg_removed_output_dir)
        edge_detection(base_bg_removed_output_dir, base_binary_output_dir)
    
    # Load existing filepaths and embeddings
    image_filenames_path = os.path.join(output_folder, 'image_filenames.txt')
    binary_image_filenames_path = os.path.join(output_folder, 'binary_image_filenames.txt')
    embeddings_path = os.path.join(output_folder, 'embeddings.npy')

    existing_image_files = []
    existing_binary_files = []
    if os.path.exists(image_filenames_path):
        with open(image_filenames_path, 'r') as f:
            existing_image_files = f.read().splitlines()
    if os.path.exists(binary_image_filenames_path):
        with open(binary_image_filenames_path, 'r') as f:
            existing_binary_files = f.read().splitlines()

    existing_embeddings = np.load(embeddings_path) if os.path.exists(embeddings_path) else None

    common_transform, binary_transform = create_transforms(config)
    dataset = InferenceDataset(
        base_crop_output_dir,
        base_binary_output_dir,
        transform=common_transform,
        transform_binary=binary_transform,
        existing_files=existing_image_files
    )

    if dataset.pairs:
        # Run inference on new images
        data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
        model = load_model(config)
        new_embeddings = []
        new_image_files = []
        new_binary_files = []

        with torch.no_grad():
            for batch, img_paths, mask_paths in data_loader:
                output = model(batch)
                new_embeddings.append(output)
                new_image_files.extend([os.path.abspath(path) for path in img_paths])
                new_binary_files.extend([os.path.abspath(path) if path != "No corresponding binary image" else path for path in mask_paths])

        new_embeddings = torch.cat(new_embeddings, dim=0)

        # Concatenate with existing embeddings
        if existing_embeddings is not None:
            all_embeddings = np.concatenate([existing_embeddings, new_embeddings.numpy()], axis=0)
        else:
            all_embeddings = new_embeddings.numpy()

        # Update filepath lists
        all_image_files = existing_image_files + new_image_files
        all_binary_files = existing_binary_files + new_binary_files

        # Save updated filepaths
        with open(image_filenames_path, 'w') as f_img, open(binary_image_filenames_path, 'w') as f_binary:
            for img_path, binary_path in zip(all_image_files, all_binary_files):
                f_img.write(f"{img_path}\n")
                f_binary.write(f"{binary_path}\n")

        # Save updated embeddings
        np.save(embeddings_path, all_embeddings)
        print(f"Updated embeddings saved to {embeddings_path}")

        # Recompute distance matrix
        if config.get("distance_metric") == "cosine":
            distance_matrix = cosine_dist(torch.from_numpy(all_embeddings), torch.from_numpy(all_embeddings))
        else:
            distance_matrix = euclidean_dist(torch.from_numpy(all_embeddings), torch.from_numpy(all_embeddings))

        # Save updated distance matrix
        np.save(os.path.join(output_folder, 'distance_matrix.npy'), distance_matrix)
        print(f"Updated distance matrix saved to {os.path.join(output_folder, 'distance_matrix.npy')}")

        print(f"{len(new_image_files)} new images processed and added.")
    else:
        print("No new images found. Embeddings and distance matrix remain unchanged.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = os.path.join(project_root, 'config_inference.json')
    run_inference(config_path)