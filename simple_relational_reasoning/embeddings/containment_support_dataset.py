import typing
import pathlib

import torch
from torchvision import transforms
from torchvision.datasets import folder

from .stimuli import NORMALIZE, DEFAULT_CANVAS_SIZE


DEFAULT_IMAGE_EXTENSION = '.png'
CONTAINMENT = 'containment'
HIGH_CONTAINMENT = 'high_containment'
BEHIND = 'behind'
SUPPORT = 'support'
SCENE_TYPES = (CONTAINMENT, HIGH_CONTAINMENT, BEHIND, SUPPORT)

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize(DEFAULT_CANVAS_SIZE),
    transforms.ToTensor(),
    NORMALIZE,
])


class ContainmentSupportDataset:
    dataset: torch.Tensor
    image_dir_path: pathlib.Path
    index_zfill: int
    extension: str
    n_configurations: int
    target_objects: list
    transform: typing.Callable
    scene_types: typing.Sequence[str]

    def __init__(self, image_dir: str, transform: typing.Callable = DEFAULT_TRANSFORM, extension: str = DEFAULT_IMAGE_EXTENSION, scene_types: typing.Sequence[str] = SCENE_TYPES):
        self.image_dir_path = pathlib.Path(image_dir)
        self.transform = transform
        self.extension = extension
        self.scene_types = scene_types

        self._create_dataset()

    def _create_dataset(self):
        path_splits = [path.name.replace(self.extension, '').split('_', 2) for path in self.image_dir_path.glob(f'*{self.extension}')]
        
        self.target_objects = list(sorted(set([path_split[0] for path_split in path_splits])))
        
        self.index_zfill = len(path_splits[0][1])
        assert all([len(path_split[1]) == self.index_zfill for path_split in path_splits])
        self.n_configurations = max([int(path_split[1]) for path_split in path_splits]) + 1
        
        assert all([path_split[2] in self.scene_types for path_split in path_splits])

        ordered_prefixes = [f'{target_object}_{str(index).zfill(self.index_zfill)}'
            for index in range(self.n_configurations)
            for target_object in self.target_objects
        ]

        dataset_tensors = []
        for prefix in ordered_prefixes:
            prefix_paths = [f'{prefix}_{scene_type}{self.extension}' for scene_type in self.scene_types]
            dataset_tensors.append(torch.stack([self.transform(folder.default_loader((self.image_dir_path / path).as_posix())) for path in prefix_paths]))

        self.dataset = torch.stack(dataset_tensors)

    def __getitem__(self, index):
        return self.dataset[index]


        