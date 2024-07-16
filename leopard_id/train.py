import json
import torch
import os
import logging
from PIL import Image

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

with open(os.path.join(project_root, "config.json"), "r") as f:
    config = json.load(f)


def setup_data_loader(config):
    root_dir_train = os.path.join(project_root, config["train_data_dir"])
    root_dir_test = os.path.join(project_root, config["test_data_dir"])
    if not os.path.isdir(root_dir_train):
        raise NotADirectoryError(f"Directory {root_dir_train} does not exist.")
    else:
        logging.info(f"Found directory {root_dir_train}")

    train_dataset = LeopardDataset(
        root_dir=root_dir_train,
        transform=transforms.Compose(
            [
                ResizeTransform(
                    width=config["resize_width"],
                    height=config["resize_height"],
                ),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
            ]
        ),
        convert=config["convert_to_RGB"]
    )

    test_dataset = LeopardDataset(
        root_dir=root_dir_test,
        transform=transforms.Compose(
            [
                ResizeTransform(
                    width=config["resize_width"],
                    height=config["resize_height"],
                ),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
            ]
        ),
        convert=config["convert_to_RGB"]
    )

    train_sampler = LeopardBatchSampler(
        train_dataset,
        batch_size=config["batch_size"],
        verbose=config["verbose"],
    )
    # DataLoader for test should have batch size equal to the number of images
    return DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4), DataLoader(
        test_dataset, batch_size=len(test_dataset)
    )


def main():
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    train_loader, test_loader = setup_data_loader(config)

    num_input_channels = next(iter(train_loader))[0].shape[1]
    logging.info(f"Number of input channels for the model: {num_input_channels}")

    model = TripletNetwork(backbone_model=config["backbone_model"],
                           input_channels=num_input_channels).to(device)
    criterion = TripletLoss(margin=config["margin"], verbose=config["verbose"])
    print(model)
    print(summary(model, (num_input_channels,
                          config["resize_height"], config["resize_width"])))

    resnet_model = train_model(
        model,
        train_loader,
        test_loader,
        device=device,
        criterion=criterion,
        config=config,
    )
    save_path = os.path.join(project_root, config["save_path"])
    torch.save(resnet_model.state_dict(), save_path)
    main_executor_visualization(resnet_model)


if __name__ == "__main__":
    main()
