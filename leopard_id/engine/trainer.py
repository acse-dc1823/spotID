import torch
import torch.optim as optim
import torch.nn as nn

# To launch TensorBoard, run tensorboard --logdir runs
from torch.utils.tensorboard import SummaryWriter
import time
import logging

from losses import TripletLoss, euclidean_dist
from metrics import (
    compute_dynamic_top_k_avg_precision,
    compute_class_distance_ratio,
    compute_top_k_rank_match_detection
)

logging.basicConfig(filename='timings.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def train_epoch(model, data_loader, optimizer, criterion, device, max_k):
    """
    Train the model for one epoch and return the average loss. Also evaluate the model
    with the chosen metrics on the training data. Do it in one function to avoid accessing
    data twice. Average metrics over all batches.
    """
    model.train()
    total_data_time = 0
    total_forward_time = 0
    total_train_eval_time = 0
    start_data_time = time.time()

    total_loss = 0
    total_precision = 0
    total_class_distance_ratio = 0
    total_match_rate = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        data_time = time.time() - start_data_time
        total_data_time += data_time

        optimizer.zero_grad()

        start_forward_time = time.time()
        outputs = model(inputs)
        forward_time = time.time() - start_forward_time
        total_forward_time += forward_time

        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            logging.error(f"Detected NaN or Inf in outputs")
            continue  

        loss = criterion(outputs, targets)

        if not loss.grad_fn:
            logging.error(f"Loss tensor with value: {loss.item()} has no grad_fn, aborting backward to avoid crash")
            continue  # Skip backward and log the issue, but avoid crashing

        loss.backward()
        optimizer.step()

        # loss is already averaged per comparison
        total_loss += loss.item()

        # start evaluation on training data
        start_eval_time = time.time()
        train_precision, train_class_distance_ratio, train_match_rate = evaluate_data(model, outputs, targets, device, max_k)
        train_eval_time = time.time() - start_eval_time
        total_train_eval_time += train_eval_time

        total_precision += train_precision
        total_class_distance_ratio += train_class_distance_ratio
        total_match_rate += train_match_rate
        start_data_time = time.time()

        torch.cuda.empty_cache()
        del inputs, outputs, loss
        torch.cuda.empty_cache()

    # Average loss per batch
    avg_loss = total_loss / len(data_loader)
    avg_precision = total_precision / len(data_loader)
    avg_class_distance_ratio = total_class_distance_ratio / len(data_loader)
    avg_match_rate = total_match_rate / len(data_loader)

    # Logging the cumulative times
    logging.info(f"Cumulative data loading time for training: {total_data_time:.4f} seconds")
    logging.info(f"Cumulative forward pass time for training: {total_forward_time:.4f} seconds")
    logging.info(f"Cumulative training evaluation time: {total_train_eval_time:.4f} seconds")

    return avg_loss, avg_precision, avg_class_distance_ratio, avg_match_rate


def evaluate_data(model, outputs, targets, device, max_k=5, verbose=False):
    """
    Evaluate the model and return the average precision, class distance
    ratio and top k match ratio using logging for verbose output. Gets called
    inside an epoch, so is returning each metric for a specific batch.
    """

    model.eval()

    with torch.no_grad():
        # Distance matrix computation
        start_time = time.time()
        dist_mat = euclidean_dist(outputs, outputs)
        time_taken_dist = time.time() - start_time

        # Precision calculation
        start_time_precision = time.time()
        batch_precision = compute_dynamic_top_k_avg_precision(
            dist_mat, targets, max_k, device
        )
        time_taken_precision = time.time() - start_time_precision

        # Class distance ratio calculation
        start_time_ratio = time.time()
        class_distance_ratio = compute_class_distance_ratio(dist_mat, targets, device)
        time_taken_ratio = time.time() - start_time_ratio

        # Top k match rate
        start_time_match_rate = time.time()
        batch_match_rate = compute_top_k_rank_match_detection(
            dist_mat, targets, max_k, device
        )
        time_taken_match_rate = time.time() - start_time_match_rate

        if verbose:
            logging.info(f"Time taken to calculate class distance: {time_taken_dist:.2f} s")
            logging.info(f"Time taken to calculate precision: {time_taken_precision:.2f} s")
            logging.info(f"Time taken to calculate class distance ratio: {time_taken_ratio:.2f} s")
            logging.info(f"Time taken to calculate top k match rate: "
                         f"{time_taken_match_rate:.2f} s")
        
        del dist_mat
    model.train()

    return batch_precision, class_distance_ratio, batch_match_rate[-1]


def evaluate_epoch_test(model, data_loader, device, max_k, optimizer, verbose=True):
    """
    Evaluation speficically for testing data inside an epoch. Accesses testing
    data, and calls evaluate_data(), which already calculates the metrics needed
    for a batch (in this case the whole test dataset).
    """

    model.eval()
    total_precision = 0
    total_class_distance_ratio = 0
    total_match_rate = 0
    data_time = 0

    start_time = time.time()

    # Check if we are using a CUDA device
    use_cuda = 'cuda' in device.type

    # Initialize the profiler with memory profiling based on the device type
    with torch.autograd.profiler.profile(use_cuda=use_cuda, profile_memory=True) as prof:
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            data_time += (time.time() - start_time)

            outputs = model(inputs)
            batch_precision, class_distance_ratio, batch_match_rate = evaluate_data(model, outputs, targets, device, max_k=max_k, verbose=verbose)
            total_precision += batch_precision
            total_class_distance_ratio += class_distance_ratio
            total_match_rate += batch_match_rate
            torch.cuda.empty_cache() if use_cuda else None
            del inputs, outputs
            torch.cuda.empty_cache() if use_cuda else None

            # Reset start time for next batch
            start_time = time.time()

    # End of profiler context, print profiling results focused on memory usage
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage" if use_cuda else "cpu_memory_usage"))

    logging.info(f"Time taken to access test data {data_time:.2f} s")

    return total_precision, total_class_distance_ratio, total_match_rate


def train_model(model, train_loader, test_loader, device, criterion, config,
                num_input_channels):
    """
    Train and evaluate the model, focusing only on the last added
    embedding layer.
    """
    if config['num_last_layers_to_train'] > 3:
        error_message = "Invalid number of layers to train specified: {}. Please set 'num_last_layers_to_train' to a maximum of 3.".format(config['num_last_layers_to_train'])
        logging.error(error_message)
        raise ValueError(error_message)

    # Freeze all pretrained layers initially
    for param in model.parameters():
        param.requires_grad = False

    trainable_params = []

    # If we have more than 3 input channels, we modify the first conv layer, and thus have to train them.
    if num_input_channels > 3:
        for param in model.backbone.backbone.conv1.parameters():
            param.requires_grad = True
        trainable_params.append({'params': model.backbone.backbone.conv1.parameters()})
        logging.info("Unfrozen parameters of first conv layer to accept"
                     "input of more than 3 channels")

    # Unfreeze and add the embedding layer parameters
    if config['num_last_layers_to_train'] >= 1:
        for param in model.embedding_layer.parameters():
            param.requires_grad = True
        trainable_params.append({'params': model.embedding_layer.parameters()})
        logging.info("Unfrozen parameters of embedding layer")

    # Unfreeze and add the last FC layer parameters if requested
    if config['num_last_layers_to_train'] >= 2 and hasattr(model.backbone.backbone, "fc"):
        for param in model.backbone.backbone.fc.parameters():
            param.requires_grad = True
        trainable_params.append({'params': model.backbone.backbone.fc.parameters()})
        logging.info("Unfrozen parameters of last FC layer")

    # Unfreeze and add the last Conv2d layer parameters if requested
    # This structure needed to generalize to other resnet models.
    if config['num_last_layers_to_train'] == 3:
        last_layer_group = list(model.backbone.backbone.layer4.children())[-1]
        if config["backbone_model"].lower() == "resnet50":
            # For ResNet50, if user requests last three, train the last two Conv2d layers in the final block
            # Do it because last conv layer is actually smaller than resnet18, so need to compensate.
            named_layers = list(last_layer_group.named_children())
            conv_layers_to_train = []
            for name, layer in reversed(named_layers):
                if isinstance(layer, nn.Conv2d):
                    conv_layers_to_train.append((name, layer))
                    if len(conv_layers_to_train) == 2:  # Only keep the last two Conv2d layers
                        break

            # Unfreeze and prepare for training
            for name, layer in conv_layers_to_train:
                for param in layer.parameters():
                    param.requires_grad = True
                last_conv_name = f"layer4.{last_layer_group._get_name()}.{name}"
                trainable_params.append({'params': layer.parameters()})
                logging.info(f"Unfrozen parameters of layer: {last_conv_name}")

        else:
            # For other architectures, proceed as usual
            last_conv = None
            last_conv_name = None
            for name, layer in reversed(list(last_layer_group.named_children())):
                if isinstance(layer, nn.Conv2d):
                    last_conv = layer
                    last_conv_name = f"layer4.{last_layer_group._get_name()}.{name}"
                    break

            if last_conv:
                for param in last_conv.parameters():
                    param.requires_grad = True
                trainable_params.append({'params': last_conv.parameters()})
                logging.info(f"Unfrozen parameters of layer: {last_conv_name}")
            else:
                logging.error("No Conv2d layer found in the last block.")

    # Setup the optimizer with only the trainable parameters
    optimizer = optim.Adam(trainable_params, lr=config["learning_rate"])

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
        "num_last_layers_trained": config['num_last_layers_to_train'],
        "train_set": config["train_data_dir"]
    }

    # Initialize logging of hyperparameters at the start of training
    writer.add_hparams(hparam_dict=hparams, metric_dict={})

    cumulative_test_eval_time = 0
    max_k = config["max_k"]

    for epoch in range(config["epochs"]):
        start_time = time.time()

        train_loss, train_precision, train_class_distance_ratio, train_match_rate = train_epoch(
            model, train_loader, optimizer, criterion, device, max_k
        )

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Precision/Train", train_precision, epoch)
        writer.add_scalar(
            "Class Distance Ratio/Train", train_class_distance_ratio, epoch
        )
        writer.add_scalar(
            "top {} match rate/Train".format(max_k), train_match_rate, epoch
        )

        if test_loader is not None:
            start_eval_test = time.time()
            test_precision, test_class_dist_ratio, test_match_rate = evaluate_epoch_test(
                model, test_loader, device, max_k=max_k, optimizer=optimizer, verbose=True
            )
            eval_test_duration = time.time() - start_eval_test
            cumulative_test_eval_time += eval_test_duration

            writer.add_scalar("Precision/Test", test_precision, epoch)
            writer.add_scalar(
                "Class Distance Ratio/Test", test_class_dist_ratio, epoch
            )
            writer.add_scalar(
            "top {} match rate/Test".format(max_k), test_match_rate, epoch
            )

        elapsed_time = time.time() - start_time

        logging.info(
            f"Epoch {epoch+1}/{config['epochs']} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Train Precision: {train_precision:.4f} - "
            f"Train Class Distance Ratio: {train_class_distance_ratio:.4f} - "
            f"Train top {max_k} match rate: {train_match_rate:.4f} - "
            f"Time: {elapsed_time:.2f} s"
        )

        if test_loader is not None:
            logging.info(
                f"Test Precision: {test_precision:.4f} - "
                f"Test Class Distance Ratio: {test_class_dist_ratio:.4f} - "
                f"Test top {max_k} match rate: {test_match_rate:.4f} - "
                f"Cumulative Test Eval Time: {cumulative_test_eval_time:.2f} s"
            )

    final_metrics = {
        "final_train_precision": train_precision,
        "final_test_precision": (
            test_precision if test_loader is not None else None
        ),
        "final_test_top {} match rate".format(max_k): (
            test_precision if test_loader is not None else None
        ),
    }

    # Update final metrics at the end of training
    writer.add_hparams(hparam_dict={}, metric_dict=final_metrics)

    writer.close()
    return model
