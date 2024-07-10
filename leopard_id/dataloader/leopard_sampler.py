import numpy as np
import random
from torch.utils.data import BatchSampler

# from torch.utils.data import DataLoader
# from torchvision import transforms

# from leopard_dataset import LeopardDataset
# from transforms import ResizeTransform


class LeopardBatchSampler(BatchSampler):
    """
    Custom batch sampler for leopards that ensures each batch contains images
    from multiple leopards. Each leopard will contribute up to a maximum of
    4 images per batch, depending on availability.

    Attributes:
        dataset (Dataset): The dataset from which to sample, expected to have
        a 'leopards' attribute listing identifiers.
        batch_size (int): The number of items in each batch.
    """

    def __init__(self, dataset, batch_size, verbose=False):
        """
        Initializes the batch sampler with dataset information, batch size.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.leopard_to_indices = self._map_leopards_to_indices()
        self.leopards = list(self.leopard_to_indices.keys())
        self.verbose = verbose

    def _map_leopards_to_indices(self):
        """
        Creates a mapping from each leopard identifier to the list of indices
        of its images in the dataset.
        This facilitates quick access to images during batch sampling.
        """
        leopard_to_indices = {}
        for index, leopard in enumerate(self.dataset.leopards):
            # First time we encounter leopard we need to instantiate list
            if leopard not in leopard_to_indices:
                leopard_to_indices[leopard] = []
            leopard_to_indices[leopard].append(index)
        return leopard_to_indices

    def __iter__(self):
        """
        An iterator that generates batches of indices from the dataset, where
        each batch contains indices of images from multiple leopards.
        The leopards are chosen with a probability proportional to the
        logarithm of the number of available images, such that leopards
        with more instances are more likely to be selected, but not
        overwhelmingly so due to the logarithm. We do this so that later
        batches aren't dominated by a single leopard (uniform probability of
        choosing leopards would actually mean leopards with more instances
        aren't consumed till the end, as they need to be chosen many times
        to consume them).

        Yields:
            list: A batch of image indices. Each batch is constrained
            by `batch_size`.

        Steps:
        1. Prepare a copy of available indices for each leopard.
        2. Calculate initial weights based on the logarithm of the number
           of available indices for active leopards.
        3. Continue to select leopards and sample images until all
           indices are exhausted:
          - Select a leopard based on the current weights.
          - Sample up to 4 images from the selected leopard.
          - Update the batch and remove the selected indices from the
            leopard's available pool.
          - If a batch reaches the specified size, yield it and
            reset for the next batch.
        4. Once all images are selected or when no more indices are available,
           finalize and yield any remaining images if they meet
           the batch size requirements.
        """
        available_indices = {
            leopard: indices.copy()
            for leopard, indices in self.leopard_to_indices.items()
        }
        batch = []
        while True:
            # Calculate weights based on the number of available
            #  indices for each leopard
            weights = [
                np.log(len(available_indices[leopard]) + 1)
                for leopard in self.leopards
                if available_indices[leopard]
            ]
            leopards_with_indices = [
                leopard
                for leopard in self.leopards
                if available_indices[leopard]
            ]

            while (
                sum(weights) > 0
            ):  # While there are still indices to process choose a leopard
                # based on the number of available indices (weighted choice)
                chosen_leopard = random.choices(
                    leopards_with_indices, weights=weights, k=1
                )[0]
                num_images = min(len(available_indices[chosen_leopard]), 4)
                selected_indices = random.sample(
                    available_indices[chosen_leopard], num_images
                )
                batch.extend(selected_indices)
                available_indices[chosen_leopard] = [
                    idx
                    for idx in available_indices[chosen_leopard]
                    if idx not in selected_indices
                ]

                if self.verbose:
                    print(
                        f"Selected indices for {chosen_leopard}: "
                        f"{selected_indices}"
                    )
                    print(
                        f"Remaining indices for {chosen_leopard}: "
                        f"{available_indices[chosen_leopard]}"
                    )

                while len(batch) >= self.batch_size:
                    if self.verbose:
                        print(f"Yielding batch of size {self.batch_size}")
                    yield batch[: self.batch_size]
                    batch = batch[
                        self.batch_size:
                    ]  # Correctly manage overflow

                # Recalculate weights after updating indices
                weights = [
                    np.log(len(available_indices[leopard]) + 1)
                    for leopard in leopards_with_indices
                    if available_indices[leopard]
                ]
                leopards_with_indices = [
                    leopard
                    for leopard in leopards_with_indices
                    if available_indices[leopard]
                ]

            if len(batch) > 0:
                if self.verbose:
                    print(f"Yielding final batch of size {len(batch)}")
                yield batch
                batch = []  # Ensure batch is cleared after yielding

            # Summing up available images
            available_images = np.sum(
                [len(indices) for indices in available_indices.values()]
            )
            if available_images == 0:
                break  # Exit loop if all indices have been used

    def __len__(self):
        """
        Provides an estimate of the number of batches per epoch based
        on total images and batch size.
        """
        # Sum up all indices available across all leopards and calculate
        # the number of batches
        total_images = np.sum(
            [len(indices) for indices in self.leopard_to_indices.values()]
        )
        return (
            total_images + self.batch_size - 1
        ) // self.batch_size  # Ceiling division for complete batches


# root_dir = '../data/crop_output'
# # Initialize the dataset
# test_dataset = LeopardDataset(
#     root_dir=root_dir,
#     transform=transforms.Compose([
#         ResizeTransform(width=512, height=256),  # Resize images
#         transforms.ToTensor()  # Convert images to tensor
#     ])
# )

# # Initialize the custom batch sampler
# test_sampler = LeopardBatchSampler(test_dataset, batch_size=32)

# # Create a DataLoader with the custom batch sampler
# test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)

# # Iterate through the DataLoader and print out batch information
# total_images = 0
# for i, (images, labels) in enumerate(test_loader):
#     print(f"\n Batch {i + 1}:")
#     print(f" - Number of images: {len(images)}")
#     print(f" - Labels: {labels}")
#     print(f" - unique labels: {set(labels)}")
#     print(f" - unique labels count: {len(set(labels))} \n \n")
#     total_images += len(images)
# print(f"Total number of images: {total_images}")
