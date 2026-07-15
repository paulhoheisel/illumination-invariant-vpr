import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from oxford_data_processing import ThresholdPairDataset, illumination_invariance_validation


class DummyDarkDrivingDataset(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        night_img = torch.ones(3, 4, 4)
        day_img = torch.ones(3, 4, 4) * 2.0
        return night_img, day_img


class IdentityEmbeddingModel(torch.nn.Module):
    def forward(self, x):
        return x.flatten(1)


def test_threshold_pair_dataset_builds_expected_groups(tmp_path):
    poses_1 = {
        'img_a': {'q': [0, 0, 0, 1], 't': np.array([0.0, 0.0, 0.0])},
        'img_b': {'q': [0, 0, 0, 1], 't': np.array([1.0, 0.0, 0.0])},
        'img_c': {'q': [0, 0, 0, 1], 't': np.array([2.5, 0.0, 0.0])},
        'img_d': {'q': [0, 0, 0, 1], 't': np.array([4.0, 0.0, 0.0])},
    }
    poses_2 = {
        'img_e': {'q': [0, 0, 0, 1], 't': np.array([0.0, 0.0, 0.0])},
        'img_f': {'q': [0, 0, 0, 1], 't': np.array([1.5, 0.0, 0.0])},
        'img_g': {'q': [0, 0, 0, 1], 't': np.array([3.0, 0.0, 0.0])},
        'img_h': {'q': [0, 0, 0, 1], 't': np.array([6.0, 0.0, 0.0])},
    }

    image_names_1 = ['img_a', 'img_b', 'img_c', 'img_d']
    image_names_2 = ['img_e', 'img_f', 'img_g', 'img_h']
    path_map_1 = {name: str(tmp_path / f'{name}.jpg') for name in image_names_1}
    path_map_2 = {name: str(tmp_path / f'{name}.jpg') for name in image_names_2}

    dataset = ThresholdPairDataset(
        thresholds=[(1, 5), (2, 10)],
        poses_dict_1=poses_1,
        poses_dict_2=poses_2,
        image_names_1=image_names_1,
        image_names_2=image_names_2,
        path_map_1=path_map_1,
        path_map_2=path_map_2,
        sample_size=2,
    )

    assert len(dataset.pairs_by_threshold) == 2
    assert len(dataset.pairs_by_threshold[(1, 5)]) >= 0
    assert dataset.threshold_labels[(1, 5)] == 0
    assert dataset.threshold_labels[(2, 10)] == 1
    assert hasattr(dataset, 'pairs')


def test_illumination_invariance_validation_returns_mean_cosine_similarity():
    dataset = DummyDarkDrivingDataset()
    night_expert = IdentityEmbeddingModel()
    day_expert = IdentityEmbeddingModel()

    mean_similarity, scores = illumination_invariance_validation(
        dataset,
        night_expert=night_expert,
        day_expert=day_expert,
        batch_size=2,
        num_workers=0,
        device=torch.device('cpu'),
    )

    assert np.isclose(mean_similarity, 1.0)
    assert len(scores) == len(dataset)
    assert np.allclose(scores, 1.0)
