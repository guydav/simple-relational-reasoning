import typing
import pathlib

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import folder
from torch.utils.data import TensorDataset
from tqdm import tqdm

from .stimuli import NORMALIZE, DEFAULT_CANVAS_SIZE


DEFAULT_IMAGE_EXTENSION = '.png'
CONTAINMENT = 'containment'
HIGH_CONTAINMENT = 'high_containment'
BEHIND = 'behind'
FAR_BEHIND = 'far_behind'
SUPPORT = 'support'
SCENE_TYPES = (CONTAINMENT, HIGH_CONTAINMENT, BEHIND, FAR_BEHIND, SUPPORT)

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize(DEFAULT_CANVAS_SIZE),
    transforms.ToTensor(),
    NORMALIZE,
])

DEFAULT_VALIDATION_PROPORTION = 0.1
DEFAULT_RANDOM_SEED = 33


class DecodingDatasets(typing.NamedTuple):
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    n_classes: int


class ContainmentSupportDataset:
    dataset: torch.Tensor
    dataset_configuration_indices: typing.List[int]
    dataset_reference_objects: typing.List[str]
    dataset_target_objects: typing.List[str]
    image_dir_path: pathlib.Path
    index_zfill: int
    extension: str
    n_configurations: int
    reference_objects: typing.List[str]
    target_objects: typing.List[str]
    transform: typing.Callable
    tqdm: bool
    scene_types: typing.Sequence[str]

    def __init__(self, image_dir: str, transform: typing.Callable = DEFAULT_TRANSFORM, extension: str = DEFAULT_IMAGE_EXTENSION, scene_types: typing.Sequence[str] = SCENE_TYPES, tqdm: bool = True):
        self.image_dir_path = pathlib.Path(image_dir)
        self.transform = transform
        self.extension = extension
        self.scene_types = scene_types
        self.tqdm = tqdm

        self.dataset_configuration_indices = []
        self.dataset_reference_objects = []
        self.dataset_target_objects = []

        self._create_dataset()

    def _create_dataset(self):
        path_splits = [path.name.replace(self.extension, '').split('_', 3) for path in self.image_dir_path.glob(f'*{self.extension}')]
        
        self.reference_objects = list(sorted(set([path_split[0] for path_split in path_splits])))
        self.target_objects = list(sorted(set([path_split[1] for path_split in path_splits])))
        
        self.index_zfill = len(path_splits[0][2])
        assert all([len(path_split[2]) == self.index_zfill for path_split in path_splits])
        self.n_configurations = max([int(path_split[2]) for path_split in path_splits]) + 1
        
        assert all([path_split[3] in self.scene_types for path_split in path_splits])

        dataset_tensors = []

        with tqdm(total=self.n_configurations * len(self.target_objects) * len(self.reference_objects)) as pbar:
            for index in range(self.n_configurations):
                for reference_object in self.reference_objects:
                    for target_object in self.target_objects:
                        prefix = f'{reference_object}_{target_object}_{str(index).zfill(self.index_zfill)}'
                        prefix_paths = [f'{prefix}_{scene_type}{self.extension}' for scene_type in self.scene_types]
                        dataset_tensors.append(torch.stack([self.transform(folder.default_loader((self.image_dir_path / path).as_posix())) for path in prefix_paths]))

                        self.dataset_configuration_indices.append(index)
                        self.dataset_reference_objects.append(reference_object)
                        self.dataset_target_objects.append(target_object)

                        pbar.update(1)

        self.dataset = torch.stack(dataset_tensors)

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)

    def _indices_to_X_y(self, indices: typing.List[int]):
        d = self.dataset[indices]
        X = d.view(-1, *d.shape[2:])
        y = torch.tensor([0, 0, 1, 1, 2]).repeat(len(indices)) # containment, behind, support
        return X, y

    def generate_decoding_datasets(self, test_target_object: typing.Optional[str] = None, test_reference_object: typing.Optional[str] = None,
        validation_proportion: float = DEFAULT_VALIDATION_PROPORTION, random_seed: int = DEFAULT_RANDOM_SEED):

        if test_target_object is None and test_reference_object is None:
            raise ValueError('test_reference_object and test_target_object cannot both be None')

        if test_target_object is not None and test_reference_object is not None:
            raise ValueError('test_reference_object and test_target_object cannot both be not None')

        if test_target_object is not None:
            train_indices = np.array([i for i in range(len(self)) if self.dataset_target_objects[i] != test_target_object])
            test_indices = np.array([i for i in range(len(self)) if self.dataset_target_objects[i] == test_target_object])

        else:
            train_indices = np.array([i for i in range(len(self)) if self.dataset_reference_objects[i] != test_reference_object])
            test_indices = np.array([i for i in range(len(self)) if self.dataset_reference_objects[i] == test_reference_object])

        rng = np.random.default_rng(random_seed)  # type: ignore
        train_val_permutation = rng.permutation(len(train_indices))  
        max_val_index = int(len(train_indices) * validation_proportion)
        validation_indices = train_indices[train_val_permutation[:max_val_index]]  
        train_indices = train_indices[train_val_permutation[max_val_index:]]

        return DecodingDatasets(
            TensorDataset(*self._indices_to_X_y(train_indices)), 
            TensorDataset(*self._indices_to_X_y(validation_indices)), 
            TensorDataset(*self._indices_to_X_y(test_indices)),
            3
        )
