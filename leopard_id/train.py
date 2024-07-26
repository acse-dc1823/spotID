import json
import torch
import os
import logging
import argparse

from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary

from model import TripletNetwork
from dataloader import LeopardDataset, LeopardBatchSampler, ResizeTransform
from engine import train_model
from losses import TripletLoss
from visualization import main_executor_visualization

logging.basicConfig(filename='logs.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

project_root = os.path.dirname(os.path.abspath(__file__))


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


def setup_data_loader(config):
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
                PixelDropout(dropout_prob=dropout_prob, apply_dropout=apply_dropout)
            ]


            binary_transforms_list.append(transforms.Normalize(mean=mean_binary_mask, std=std_binary_mask))
        
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


def main(config_file="config.json"):
    config_path = os.path.join(project_root, config_file)

    with open(config_path, "r") as f:
        config = json.load(f)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    train_loader, test_loader = setup_data_loader(config)

    num_input_channels = next(iter(train_loader))[0].shape[1]
    logging.info(f"Number of input channels for the model: {num_input_channels}")

    model = TripletNetwork(backbone_model=config["backbone_model"],
                           num_dims=config["number_embedding_dimensions"],
                           input_channels=num_input_channels).to(device)
    print(model)
    print(summary(model, (num_input_channels,
                          config["resize_height"], config["resize_width"])))

    resnet_model = train_model(
        model,
        train_loader,
        test_loader,
        device=device,
        config=config,
        num_input_channels=num_input_channels
    )
    save_path = os.path.join(project_root, config["save_path"])
    torch.save(resnet_model.state_dict(), save_path)
    main_executor_visualization(resnet_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the triplet network training with an optional configuration file.")
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the configuration JSON file. Default is 'config.json'.")
    args = parser.parse_args()
    main(args.config_file)
