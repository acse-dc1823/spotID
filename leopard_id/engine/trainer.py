import torch
import torch.optim as optim

# To launch TensorBoard, run tensorboard --logdir runs
from torch.utils.tensorboard import SummaryWriter
import time
import logging

from losses import TripletLoss, euclidean_dist
from metrics import (
    compute_dynamic_k_avg_precision,
    compute_class_distance_ratio,
)

logging.basicConfig(filename='timings.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def train_epoch(model, data_loader, optimizer, criterion, device):
    """
    Train the model for one epoch and return the average loss.
    """
    model.train()
    total_loss = 0
    total_data_time = 0
    total_forward_time = 0

    for inputs, targets in data_loader:
        start_data_time = time.time()
        inputs, targets = inputs.to(device), targets.to(device)
        data_time = time.time() - start_data_time
        total_data_time += data_time

        optimizer.zero_grad()

        start_forward_time = time.time()
        outputs = model(inputs)
        forward_time = time.time() - start_forward_time
        total_forward_time += forward_time

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # loss is already averaged per comparison
        total_loss += loss.item()

    # Average loss per batch
    avg_loss = total_loss / len(data_loader)

    # Logging the cumulative times
    logging.info(f"Cumulative data loading time for training: {total_data_time:.4f} seconds")
    logging.info(f"Cumulative forward pass time for training: {total_forward_time:.4f} seconds")

    return avg_loss


#TODO: Change from one epoch to it being done in the train above, as accessing the data is what seems to be the bottleneck.
def evaluate_epoch(model, data_loader, device, max_k=5, verbose=False):
    """
    Evaluate the model and return the average precision and class distance
    ratio using logging for verbose output.
    """

    model.eval()
    total_precision = 0
    total_class_distance_ratio = 0

    # Cumulative times
    cumulative_data_access_time = 0
    cumulative_output_time = 0
    cumulative_distance_time = 0
    cumulative_precision_time = 0
    cumulative_ratio_time = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            general_time = time.time()

            # Data access time measurement
            inputs, targets = inputs.to(device), targets.to(device)
            time_access = time.time()
            data_access_time = time_access - general_time
            cumulative_data_access_time += data_access_time

            if verbose:
                logging.info(f"Time to access data: {data_access_time:.4f} seconds")

            # Output computation time measurement
            outputs = model(inputs)
            time_outputs = time.time()
            output_time = time_outputs - time_access
            cumulative_output_time += output_time

            if verbose:
                logging.info(f"Time to get outputs: {output_time:.4f} seconds")

            # Distance matrix computation
            start_time = time.time()
            dist_mat = euclidean_dist(outputs, outputs)
            time_taken_dist = time.time() - start_time
            cumulative_distance_time += time_taken_dist

            if verbose:
                logging.info(f"Time taken to calculate class distance: {time_taken_dist:.2f} s")

            # Precision calculation
            start_time_precision = time.time()
            batch_precision = compute_dynamic_k_avg_precision(dist_mat, targets, max_k, device)
            time_taken_precision = time.time() - start_time_precision
            cumulative_precision_time += time_taken_precision

            if verbose:
                logging.info(f"Time taken to calculate precision: {time_taken_precision:.2f} s")

            # Class distance ratio calculation
            start_time_ratio = time.time()
            class_distance_ratio = compute_class_distance_ratio(dist_mat, targets, device)
            time_taken_ratio = time.time() - start_time_ratio
            cumulative_ratio_time += time_taken_ratio

            if verbose:
                logging.info(f"Time taken to calculate class distance ratio: {time_taken_ratio:.2f} s")

            total_precision += batch_precision
            total_class_distance_ratio += class_distance_ratio

    average_precision = total_precision / len(data_loader)
    average_class_distance_ratio = total_class_distance_ratio / len(data_loader)

    # Logging cumulative times after the epoch
    logging.info(f"Cumulative data access time: {cumulative_data_access_time:.4f} seconds")
    logging.info(f"Cumulative output computation time: {cumulative_output_time:.4f} seconds")
    logging.info(f"Cumulative distance matrix computation time: {cumulative_distance_time:.4f} seconds")
    logging.info(f"Cumulative precision computation time: {cumulative_precision_time:.4f} seconds")
    logging.info(f"Cumulative ratio computation time: {cumulative_ratio_time:.4f} seconds")

    return average_precision, average_class_distance_ratio


def train_model(model, train_loader, test_loader, device, criterion, config):
    """
    Train and evaluate the model, focusing only on the last added
    embedding layer.
    """
    # Freeze all pretrained layers
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Only parameters of the embedding layer are trainable
    optimizer = optim.Adam(
        model.embedding_layer.parameters(), lr=config["learning_rate"]
    )

    criterion = TripletLoss(verbose=config["verbose"], margin=config["margin"])
    writer = SummaryWriter()  # TensorBoard summary writer initialized here

    model.to(device)

    hparams = {
        "lr": config["learning_rate"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "device": str(device),
        "margin": criterion.margin if hasattr(criterion, "margin") else -1,
        "backbone_model": config["backbone_model"],
        "max_k": config["max_k"],
    }

    # Initialize logging of hyperparameters at the start of training
    writer.add_hparams(hparam_dict=hparams, metric_dict={})

    cumulative_train_eval_time = 0
    cumulative_test_eval_time = 0

    for epoch in range(config["epochs"]):
        start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        start_eval_train = time.time()
        train_precision, train_class_distance_ratio = evaluate_epoch(
            model, train_loader, device, max_k=config["max_k"]
        )
        eval_train_duration = time.time() - start_eval_train
        cumulative_train_eval_time += eval_train_duration

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Precision/Train", train_precision, epoch)
        writer.add_scalar(
            "Class Distance Ratio/Train", train_class_distance_ratio, epoch
        )

        if test_loader is not None:
            start_eval_test = time.time()
            test_precision, test_class_dist_ratio = evaluate_epoch(
                model, test_loader, device, verbose=True, max_k=config["max_k"]
            )
            eval_test_duration = time.time() - start_eval_test
            cumulative_test_eval_time += eval_test_duration

            writer.add_scalar("Precision/Test", test_precision, epoch)
            writer.add_scalar(
                "Class Distance Ratio/Test", test_class_dist_ratio, epoch
            )

        elapsed_time = time.time() - start_time

        logging.info(
            f"Epoch {epoch+1}/{config['epochs']} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Train Precision: {train_precision:.4f} - "
            f"Train Class Distance Ratio: {train_class_distance_ratio:.4f} - "
            f"Time: {elapsed_time:.2f} s"
        )
        logging.info(f"Cumulative Train Eval Time: "
                     f"{cumulative_train_eval_time:.2f} s")

        if test_loader is not None:
            logging.info(
                f"Test Precision: {test_precision:.4f} - "
                f"Test Class Distance Ratio: {test_class_dist_ratio:.4f} - "
                f"Cumulative Test Eval Time: {cumulative_test_eval_time:.2f} s"
            )

    final_metrics = {
        "final_train_precision": train_precision,
        "final_test_precision": (
            test_precision if test_loader is not None else None
        ),
    }

    # Update final metrics at the end of training
    writer.add_hparams(hparam_dict={}, metric_dict=final_metrics)

    writer.close()
    return model
