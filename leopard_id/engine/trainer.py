import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR

# To launch TensorBoard, run tensorboard --logdir runs
from torch.utils.tensorboard import SummaryWriter
import time
import logging

from losses import TripletLoss, euclidean_dist
from model import CosFace, cosine_dist
from metrics import (
    compute_dynamic_top_k_avg_precision,
    compute_class_distance_ratio,
    compute_top_k_rank_match_detection
)

logging.basicConfig(filename='timings.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def train_epoch(model, data_loader, optimizer, criterion, device, max_k, epoch, method="triplet", classifier=None):
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

        if method == "triplet": # add this later
            loss = criterion(outputs, targets, epoch)
        
        else: # For now, only else is cosface loss. Might have to change later
            # For cosface, need to pass the logits through the cosface layer
            # to get the classification loss
            outputs_cosface = classifier(outputs, targets)
            loss = criterion(outputs_cosface, targets)

        if not loss.grad_fn:
            logging.error(f"Loss tensor with value: {loss.item()} has no grad_fn, aborting backward to avoid crash")
            continue  # Skip backward and log the issue, but avoid crashing

        loss.backward()
        optimizer.step()

        # loss is already averaged per comparison
        total_loss += loss.item()

        # start evaluation on training data
        start_eval_time = time.time()
        train_precision, train_class_distance_ratio, train_match_rate, _ = evaluate_data(model, outputs, targets,
                                                                                      device,
                                                                                      method=method,
                                                                                      max_k=max_k)
        train_eval_time = time.time() - start_eval_time
        total_train_eval_time += train_eval_time

        total_precision += train_precision
        total_class_distance_ratio += train_class_distance_ratio
        total_match_rate += train_match_rate
        start_data_time = time.time()

        torch.cuda.empty_cache()
        del inputs, outputs, loss, targets
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


def evaluate_data(model, outputs, targets, device, method="triplet", max_k=5, verbose=False):
    """
    Evaluate the model and return the average precision, class distance
    ratio and top k match ratio using logging for verbose output. Gets called
    inside an epoch, so is returning each metric for a specific batch.
    """

    model.eval()

    with torch.no_grad():
        # Distance matrix computation
        start_time = time.time()
        if method == "triplet":
            dist_mat = euclidean_dist(outputs, outputs)
        else: # For now, only else is cosface loss. Might have to change later
            dist_mat = cosine_dist(outputs, outputs)
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
        
    model.train()

    return batch_precision, class_distance_ratio, batch_match_rate, dist_mat


def evaluate_epoch_test(model, data_loader, device, max_k, method="triplet",
                        verbose=True):
    """
    Evaluate specifically for testing data inside an epoch. Collects outputs across
    batches and then evaluates them all together to compute metrics for the entire test dataset.
    """
    model.eval()
    all_outputs = []
    all_targets = []

    data_time = 0

    use_cuda = 'cuda' in device.type

    with torch.no_grad():  # Disable gradient computation to save memory and computations
        for inputs, targets in data_loader:
            batch_start_time = time.time()
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass for current batch
            outputs = model(inputs)
            
            # Store outputs and targets to compute metrics later
            all_outputs.append(outputs.detach())  # Detach outputs to avoid saving computation graph
            all_targets.append(targets.detach())

            data_time += (time.time() - batch_start_time)

            # Clean up to prevent memory leaks
            del inputs, outputs, targets
            if use_cuda:
                torch.cuda.empty_cache()

        # Concatenate all collected outputs and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Evaluate the entire dataset at once
        total_precision, class_distance_ratio, total_match_rate, dist_mat = evaluate_data(
            model, all_outputs, all_targets, device, method=method, max_k=max_k, verbose=verbose
        )

        logging.info("Cosine distance matrix for testing data, top 10 exemplars: ")
        logging.info(dist_mat[:10, :10])
        logging.info(all_targets[:10])

    logging.info(f"Time taken to access test data {data_time:.2f} s")

    return total_precision, class_distance_ratio, total_match_rate


def train_model(model, train_loader, test_loader, device,
                config, num_input_channels):
    """
    Train and evaluate the model, focusing only on the last added
    embedding layer.
    """
    method = config["method"]
    
    if config['num_last_layers_to_train'] > 3:
        error_message = "Invalid number of layers to train specified: {}. Please set 'num_last_layers_to_train' to a maximum of 3.".format(config['num_last_layers_to_train'])
        logging.error(error_message)
        raise ValueError(error_message)

    if config.get('train_all_layers', False):
        logging.info("All layers are set to be trainable.")
        for param in model.parameters():
            param.requires_grad = True
    else:
        # Freeze all pretrained layers initially
        for param in model.parameters():
            param.requires_grad = False

    trainable_params = []

    def print_layer_status(model):
        for name, module in model.named_modules():
            if any(param.requires_grad for param in module.parameters()):
                logging.info(f"Layer {name} is unfrozen for training.")
            else:
                logging.info(f"Layer {name} is frozen.")

    # If we have more than 3 input channels, we modify the first conv layer, and thus have to train them.
    if num_input_channels > 3:
        for param in model.final_backbone.conv1.parameters():
            param.requires_grad = True
        trainable_params.append({'params': model.final_backbone.conv1.parameters()})
        logging.info("Unfrozen parameters of first conv layer to accept"
                     "input of more than 3 channels")

    # Unfreeze and add the embedding layer parameters
    if config['num_last_layers_to_train'] >= 1:
        for param in model.embedding_layer.parameters():
            param.requires_grad = True
        trainable_params.append({'params': model.embedding_layer.parameters()})
        logging.info("Unfrozen parameters of embedding layer")

    # Unfreeze and add the last FC layer parameters if requested
    if config['num_last_layers_to_train'] >= 2 and hasattr(model.final_backbone, "fc"):
        for param in model.final_backbone.fc.parameters():
            param.requires_grad = True
        trainable_params.append({'params': model.final_backbone.fc.parameters()})
        logging.info("Unfrozen parameters of last FC layer")

    # Unfreeze and add the last Conv2d layer parameters if requested
    # This structure needed to generalize to other resnet models.
    if config['num_last_layers_to_train'] == 3:
        last_layer_group = list(model.final_backbone.layer4.children())[-1]
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

    print_layer_status(model)

    # Setup the optimizer with only the trainable parameters
    optimizer = optim.Adam(trainable_params, lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.7)


    if method == "triplet":
        criterion = TripletLoss(verbose=config["verbose"], margin=config["margin"])
    else:
        criterion = CrossEntropyLoss()

    writer = SummaryWriter()  # TensorBoard summary writer initialized here

    model.to(device)

    hparams = {
        "lr": config["learning_rate"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "device": str(device),
        "margin": config["margin"],
        "backbone_model": config["backbone_model"],
        "max_k": config["max_k"],
        "num_last_layers_trained": config['num_last_layers_to_train'],
        "train_all_layers": config["train_all_layers"],
        "train_set": config["train_data_dir"]
    }

    # Initialize logging of hyperparameters at the start of training
    writer.add_hparams(hparam_dict=hparams, metric_dict={})

    cumulative_test_eval_time = 0
    max_k = config["max_k"]

    if method == "cosface":
        num_classes = max(train_loader.dataset.label_to_index.values()) + 1
        logging.info(f"Setting up CosFace loss: Num classes for cosface: {num_classes}")
        classifier = CosFace(config["number_embedding_dimensions"], num_classes, 
                             margin=config["margin"]).to(device)
    else:
        logging.info("Setting up Triplet loss")
        classifier = None

    for epoch in range(config["epochs"]):
        start_time = time.time()

        train_loss, train_precision, train_class_distance_ratio, train_match_rate = train_epoch(
            model, train_loader, optimizer, criterion, device, max_k, epoch, method=method, classifier=classifier
        )

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Precision/Train", train_precision, epoch)
        writer.add_scalar(
            "Class Distance Ratio/Train", train_class_distance_ratio, epoch
        )
        writer.add_scalar(
            "top {} match rate/Train".format(max_k), train_match_rate[-1], epoch
        )

        if test_loader is not None:
            start_eval_test = time.time()
            test_precision, test_class_dist_ratio, test_match_rate = evaluate_epoch_test(
                model, test_loader, device, method=method, max_k=max_k, verbose=True)
            eval_test_duration = time.time() - start_eval_test
            cumulative_test_eval_time += eval_test_duration

            writer.add_scalar("Precision/Test", test_precision, epoch)
            writer.add_scalar(
                "Class Distance Ratio/Test", test_class_dist_ratio, epoch
            )
            writer.add_scalar(
            "top {} match rate/Test".format(max_k), test_match_rate[-1], epoch
            )

        elapsed_time = time.time() - start_time

        logging.info(
            f"Epoch {epoch+1}/{config['epochs']} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Train Precision: {train_precision:.4f} - "
            f"Train Class Distance Ratio: {train_class_distance_ratio:.4f} - "
            f"Train top {max_k} match rate: {train_match_rate[-1]:.4f} - "
            f"Time: {elapsed_time:.2f} s"
        )

        if test_loader is not None:
            logging.info(
                f"Test Precision: {test_precision:.4f} - "
                f"Test Class Distance Ratio: {test_class_dist_ratio:.4f} - "
                f"Test top {max_k} match rate: {test_match_rate[-1]:.4f} - "
                f"Cumulative Test Eval Time: {cumulative_test_eval_time:.2f} s"
            )
        scheduler.step()

    _, _, final_test_match_rate = evaluate_epoch_test(
        model, test_loader, device, method=method, max_k=20, verbose=True)
    for k in range(final_test_match_rate.size(0)):
        writer.add_scalar("top k final match rate/Test Final",
                          final_test_match_rate[k], k)
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
