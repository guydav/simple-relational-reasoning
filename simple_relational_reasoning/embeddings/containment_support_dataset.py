import typing
import pathlib

import torch
from torchvision import transforms
from torchvision.datasets import folder
from tqdm import tqdm

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

        self.dataset_bowl_colors = []
        self.dataset_configuration_indices = []
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


        