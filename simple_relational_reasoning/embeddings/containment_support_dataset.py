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

HABITUATION = 'habituation'
SAME_RELATION = 'same'
DIFFERENT_RELATION = 'different'
QUINN_SCENE_TYPES = (HABITUATION, SAME_RELATION, DIFFERENT_RELATION)

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
    test_configurations: typing.List[typing.Dict[str, typing.Union[int, str]]]


class ContainmentSupportDataset:
    dataset: torch.Tensor
    dataset_configuration_indices: typing.List[int]
    dataset_habituation_target_objects: typing.List[str]
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
    shuffle_habituation_stimuli: bool

    def __init__(self, image_dir: str, shuffle_habituation_stimuli: bool = False, 
        transform: typing.Callable = DEFAULT_TRANSFORM, 
        extension: str = DEFAULT_IMAGE_EXTENSION, 
        scene_types: typing.Sequence[str] = SCENE_TYPES, tqdm: bool = True, random_seed: int = DEFAULT_RANDOM_SEED):

        self.image_dir_path = pathlib.Path(image_dir)
        self.shuffle_habituation_stimuli = shuffle_habituation_stimuli
        self.transform = transform
        self.extension = extension
        self.scene_types = scene_types
        self.tqdm = tqdm

        self.dataset_configuration_indices = []
        self.dataset_reference_objects = []
        self.dataset_target_objects = []
        self.dataset_habituation_target_objects = []

        self.rng = np.random.default_rng(random_seed)  # type: ignore

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

        default_target_indices = np.arange(len(self.target_objects))

        with tqdm(total=self.n_configurations * len(self.target_objects) * len(self.reference_objects)) as pbar:
            for index in range(self.n_configurations):
                for reference_object in self.reference_objects:
                    current_reference_stimuli = []
                    for target_object in self.target_objects:
                        prefix = f'{reference_object}_{target_object}_{str(index).zfill(self.index_zfill)}'
                        prefix_paths = [f'{prefix}_{scene_type}{self.extension}' for scene_type in self.scene_types]
                        current_reference_stimuli.append(torch.stack([self.transform(folder.default_loader((self.image_dir_path / path).as_posix())) for path in prefix_paths]))

                        self.dataset_configuration_indices.append(index)
                        self.dataset_reference_objects.append(reference_object)
                        self.dataset_target_objects.append(target_object)

                        pbar.update(1)

                    current_reference_tensor = torch.stack(current_reference_stimuli)

                    if self.shuffle_habituation_stimuli:
                        perm = self.rng.permutation(len(self.target_objects))
                        while (perm == default_target_indices).any():
                            perm = self.rng.permutation(len(self.target_objects))
                        
                        current_reference_tensor[:, 0] = current_reference_tensor[perm, 0]
                        self.dataset_habituation_target_objects.extend([self.target_objects[i] for i in perm])

                    else:
                        self.dataset_habituation_target_objects.extend(self.target_objects)

                    dataset_tensors.append(current_reference_tensor) 

        self.dataset = torch.concat(dataset_tensors)

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)

    def _indices_to_X_y(self, indices: typing.List[int]):
        d = self.dataset[indices]
        X = d.view(-1, *d.shape[2:])
        y = torch.tensor([0, 0, 1, 1, 2]).repeat(len(indices)) # containment, behind, support
        return X, y

    def _split_indices(self, indices: np.ndarray, proportion: float):
        index_permutation = self.rng.permutation(len(indices))  
        max_split_index = int(len(indices) * proportion)
        return indices[index_permutation[max_split_index:]], indices[index_permutation[:max_split_index]]  

    def generate_decoding_datasets(self, test_target_object: typing.Optional[str] = None, test_reference_object: typing.Optional[str] = None,
        test_proportion: typing.Optional[float] = None, validation_proportion: float = DEFAULT_VALIDATION_PROPORTION, test_seed: typing.Optional[int] = None):

        if test_seed is not None:
            self.rng = np.random.default_rng(test_seed)

        if test_target_object is None and test_reference_object is None and test_proportion is None:
            raise ValueError('test_reference_object, test_target_object, and test_proportion cannot all be None')

        test_index_set = set()

        if test_target_object is not None:
            test_index_set.update(i for i in range(len(self)) if self.dataset_target_objects[i] == test_target_object)

        if test_reference_object is not None:
            test_index_set.update(i for i in range(len(self)) if self.dataset_reference_objects[i] == test_reference_object)
        
        if test_proportion is not None:  # test_proportion is not None
            test_proportion = typing.cast(float, test_proportion)  
            unique_configurations = np.array(list(set(self.dataset_configuration_indices)))  
            _, test_configurations = self._split_indices(unique_configurations, test_proportion)
            test_index_set.update(i for i in range(len(self)) if self.dataset_configuration_indices[i] in test_configurations)

        train_indices = []
        test_indices = []
        for i in range(len(self)):
            if i in test_index_set:
                test_indices.append(i)
            else:
                train_indices.append(i)

        test_configurations = [dict(configuration_index=self.dataset_configuration_indices[i], reference_object=self.dataset_reference_objects[i], target_object=self.dataset_target_objects[i]) for i in test_indices]

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        

        train_indices, validation_indices = self._split_indices(train_indices, validation_proportion)
        
        return DecodingDatasets(
            TensorDataset(*self._indices_to_X_y(train_indices)), 
            TensorDataset(*self._indices_to_X_y(validation_indices)), 
            TensorDataset(*self._indices_to_X_y(test_indices)),  # type: ignore
            n_classes=3,
            test_configurations=test_configurations,
        )
