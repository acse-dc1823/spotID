# author: David Colomer Matachana
# GitHub username: acse-dc1823

import pytest
from leopard_id.dataloader import LeopardBatchSampler

@pytest.fixture
def sample_dataset():
    class MockDataset:
        def __init__(self):
            self.leopards = ['leopard1', 'leopard1', 'leopard2', 'leopard2', 'leopard2',
                             'leopard3', 'leopard4', 'leopard4', 'leopard5', 'leopard6',
                             'leopard6', 'leopard6', 'leopard6', 'leopard6', 'leopard6',
                             'leopard7', 'leopard7', 'leopard7', 'leopard7', 'leopard8',
                             'leopard8', 'leopard8', 'leopard8', 'leopard8', 'leopard8',
                             'leopard8']

    dataset = MockDataset()
    return dataset


@pytest.fixture
def leopard_sampler(sample_dataset):
    sampler = LeopardBatchSampler(sample_dataset, batch_size=4)
    sampler.leopard_to_indices = {
        'leopard1': [0, 1],
        'leopard2': [2, 3, 4],
        'leopard3': [5],
        'leopard4': [6, 7],
        'leopard5': [8],
        'leopard6': [9, 10, 11, 12, 13, 14],
        'leopard7': [15, 16, 17, 18],
        'leopard8': [19, 20, 21, 22, 23, 24, 25],
    }
    return sampler


@pytest.fixture
def simple_dataloader(leopard_sampler):
    # Assuming the data loader simply iterates over batches from the sampler
    class SimpleDataLoader:
        def __init__(self, sampler):
            self.sampler = sampler

        def __iter__(self):
            for batch in iter(self.sampler):
                yield batch

    return SimpleDataLoader(leopard_sampler)


def test_exhaustive_use_of_images(simple_dataloader, sample_dataset):
    """
    Test that all the images in the dataset are loaded by the dataloader
    just once, no more, no less.
    """
    total_images_counted = 0
    for batch in simple_dataloader:
        total_images_counted += len(batch)  # Count indices in each batch

    expected_image_count = len(sample_dataset.leopards)
    assert total_images_counted == expected_image_count, \
        "The total number of images processed does not match the dataset size"


def test_batch_distribution(leopard_sampler):
    """
    Here, we test that the batch sampler samples according to the 
    expected distribution of leopards.
    """
    # Generate a significant number of batches to check distribution
    leopard_counts = {key: 0 for key in leopard_sampler.leopard_to_indices}
    num_batches = 400
    for _ in range(num_batches):
        batch = next(iter(leopard_sampler))
        for idx in batch:
            for leopard, indices in leopard_sampler.leopard_to_indices.items():
                if idx in indices:
                    leopard_counts[leopard] += 1
                    break

    # Ensure each leopard appears roughly proportional to its available indices
    total_images = sum(len(indices) for indices in leopard_sampler.leopard_to_indices.values())
    for leopard, count in leopard_counts.items():
        expected_count = len(leopard_sampler.leopard_to_indices[leopard]) * num_batches * 4 / total_images
        assert abs(count / num_batches - expected_count / num_batches) < 0.3


def test_batch_sizes(leopard_sampler):
    batches = list(iter(leopard_sampler))
    for batch in batches[:-1]:
        assert len(batch) == 4
    assert len(batches[-1]) <= 4
