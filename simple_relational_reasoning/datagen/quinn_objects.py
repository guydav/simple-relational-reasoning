from abc import abstractmethod
from collections import defaultdict
import itertools

import numpy as np
import torch


"""
Idea:
We have a dataset generator object that:
- receives a random seed
- generates a single training set (inherits from `torch.utils.data.Dataset`)
- generates one or more test sets (also inherit from the same)

Questions:
- Do we ever have a y size? That is, height?
- How systematic do I want to be with held out locations?
    - For example, with the grid, do I hold out a location in one row or one column?
    - Same with the reference object locations, do I hold out systematically or randomly? 
- Do we care about testing "neither"? Or not really?
- Do we want distractor objects? Or are we okay without them?
    Start without, think about whether or not we need to add.
- Dataset size
    Aim for 4k-16k 
- Different train-test objects
    Use 1 for train and the same for test
    Use 1 for train and one for test
    Use 4 for train and a unique one for test
- Implement PrediNet at some point
- Pre-trained models:
    Quinn-like stimuli, per-trained models, embeding similarities in triplets  

Try both:
- Do we represent objects with a size? Or as a collection of size one objects?
- Do we always also introduce a "neither" class? 
    Try with and without, see what happens

"""


class ObjectGenerator:
    def __init__(self, seed, reference_object_length, target_object_length=1, n_reference_types=1,
                 n_train_target_types=1, n_test_target_types=0, n_non_type_fields=None, dtype=torch.float):
        self.seed = seed
        self.reference_object_length = reference_object_length
        self.target_object_length = target_object_length
        self.n_reference_types = n_reference_types
        self.n_train_target_types = n_train_target_types
        self.n_test_target_types = n_test_target_types
        self.n_types = self.n_reference_types + self.n_train_target_types + self.n_test_target_types
        self.n_non_type_fields = n_non_type_fields
        self.dtype = dtype

        self.rng = np.random.default_rng(self.seed)

    def reference_object(self, x, y, train=True):
        raise NotImplementedError()

    def target_object(self, x, y, train=True):
        raise NotImplementedError()

    def _sample_type(self, target=False, train=True):
        if not target:  # reference
            if self.n_reference_types <= 1:
                return 0

            return self.rng.integers(self.n_reference_types)

        if train or self.n_test_target_types == 0:
            if self.n_train_target_types <= 1:
                return self.n_reference_types  # 0-based, so this is the first index

            return self.rng.integers(self.n_reference_types, self.n_reference_types + self.n_train_target_types)

        # test and we have test only target types
        if self.n_test_target_types == 1:
            return self.n_reference_types + self.n_train_target_types  # 0-based, so this is the first one

        return self.rng.integers(self.n_reference_types + self.n_train_target_types, self.n_types)

    def _to_one_hot(self, n, n_types):
        one_hot = np.zeros(n_types)
        one_hot[n] = 1
        return one_hot

    def _sample_type_one_hot(self, target=False, train=True):
        return self._to_one_hot(self._sample_type(target, train), self.n_types)

    def get_type_slice(self):
        return slice(self.n_non_type_fields, self.n_non_type_fields + self.n_types)

    def get_position_slice(self):
        # TODO: should this include the length/size of the object?
        return slice(0, 2)

class ObjectGeneratorWithSize(ObjectGenerator):
    def __init__(self, seed, reference_object_length, target_object_length=1, n_reference_types=1,
                 n_train_target_types=1, n_test_target_types=0, dtype=torch.float):
        super(ObjectGeneratorWithSize, self).__init__(seed, reference_object_length, target_object_length,
                                                      n_reference_types, n_train_target_types, n_test_target_types,
                                                      n_non_type_fields=3, dtype=dtype)

    def reference_object(self, x, y, train=True):
        return torch.tensor([x, y, self.reference_object_length, *self._sample_type_one_hot(False, train)],
                            dtype=self.dtype).unsqueeze(0)

    def target_object(self, x, y, train=True):
        return torch.tensor([x, y, self.target_object_length, *self._sample_type_one_hot(True, train)],
                            dtype=self.dtype).unsqueeze(0)


class ObjectGeneratorWithoutSize(ObjectGenerator):
    def __init__(self, seed, reference_object_length, target_object_length=1, n_reference_types=1,
                 n_train_target_types=1, n_test_target_types=0, dtype=torch.float):
        super(ObjectGeneratorWithoutSize, self).__init__(seed, reference_object_length, target_object_length,
                                                         n_reference_types, n_train_target_types, n_test_target_types,
                                                         n_non_type_fields=2, dtype=dtype)

    def reference_object(self, x, y, train=True):
        object_type = self._sample_type_one_hot(False, train)
        return torch.cat([torch.tensor([x + j, y, *object_type],
                                       dtype=self.dtype).unsqueeze(0)
                          for j in range(self.reference_object_length)])

    def target_object(self, x, y, train=True):
        object_type = self._sample_type_one_hot(True, train)
        return torch.cat([torch.tensor([x + j, y, *object_type],
                                       dtype=self.dtype).unsqueeze(0)
                          for j in range(self.target_object_length)])


class DiagonalObjectGeneratorWithoutSize(ObjectGeneratorWithoutSize):
    def __init__(self, seed, reference_object_length, target_object_length=1, n_reference_types=1,
                 n_train_target_types=1, n_test_target_types=0, dtype=torch.float):
        super(DiagonalObjectGeneratorWithoutSize, self).__init__(seed, reference_object_length, target_object_length,
                                                         n_reference_types, n_train_target_types, n_test_target_types,
                                                         dtype=dtype)

    def reference_object(self, x, y, train=True):
        object_type = self._sample_type_one_hot(False, train)
        return torch.cat([torch.tensor([x + j, y + j, *object_type],
                                       dtype=self.dtype).unsqueeze(0)
                          for j in range(self.reference_object_length)])

    def target_object(self, x, y, train=True):
        object_type = self._sample_type_one_hot(True, train)
        return torch.cat([torch.tensor([x + j, y, *object_type],
                                       dtype=self.dtype).unsqueeze(0)
                          for j in range(self.target_object_length)])



class MinimalDataset(torch.utils.data.Dataset):
    def __init__(self, objects, labels, object_generator):
        super(MinimalDataset, self).__init__()

        if not isinstance(objects, torch.Tensor):
            objects = torch.stack(objects)

        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)

        self.objects = objects
        self.labels = labels
        self.object_generator = object_generator

    def __getitem__(self, item):
        return self.objects[item], self.labels[item]

    def __len__(self):
        return self.objects.shape[0]

    def get_object_size(self):
        return self.objects.shape[-1]

    def get_num_objects(self):
        return self.objects.shape[1]

    def get_num_classes(self):
        return len(self.labels.unique())

    def subsample(self, rng, prop=None, total=None):
        if prop is None and total is None:
            raise ValueError(f'Mush provide MinimalDataset.subsample() with either prop or total')

        if prop is None:  # total is not None
            if total >= len(self) or total <= 0:
                raise ValueError(f'Total must be between 1 and the length ({len(self)}, but received {total}')
            prop = total / len(self)

        if prop <= 0 or prop >= 1:
            raise ValueError(f'Prop must be between 0 and 1, but received {prop}')

        unique_labels, counts = self.labels.unique(return_counts=True)
        float_counts = counts * prop
        count_per_label = torch.round(float_counts)
        diffs = float_counts - count_per_label
        while count_per_label.sum().item() < total:
            max_diff_index = diffs.argmax().item()
            count_per_label[max_diff_index] += 1
            diffs[max_diff_index] = -10

        while count_per_label.sum().item() > total:
            min_diff_index = diffs.argmin().item()
            count_per_label[min_diff_index] -= 1
            diffs[min_diff_index] = 10

        count_per_label = count_per_label.to(torch.int)
        indices_per_label = [torch.where(self.labels == l)[0] for l in unique_labels]
        sample_indices_per_label = [rng.permutation(indices)[:count]
                                    for indices, count in zip(indices_per_label, count_per_label)]

        if hasattr(self, 'spatial_objects'):
            self.spatial_objects = torch.cat([self.spatial_objects[indices] for indices in sample_indices_per_label])

        else:
            self.objects = torch.cat([self.objects[indices] for indices in sample_indices_per_label])

        self.labels = torch.cat([self.labels[indices] for indices in sample_indices_per_label])


class MinimalSpatialDataset(MinimalDataset):
    def __init__(self, objects, labels, object_generator, x_max, y_max, position_indices=(0, 1)):
        super(MinimalSpatialDataset, self).__init__(objects, labels, object_generator)

        D, N, O = self.objects.shape

        position_shape = (x_max, y_max)
        spatial_shape = (D, O, *position_shape)
        spatial_objects = torch.zeros(spatial_shape, dtype=self.objects.dtype)

        # TODO: handle this better for objects with sizes

        for ex_index in range(D):
            position_lists = [self.objects[ex_index, :, index].long()
                              for index in position_indices]

            # if torch.any(position_lists[0] > 24) or torch.any(position_lists[1] > 24) or self.objects[ex_index].max() > 24:
                # print('*' * 33 + ' FOUND ' + '*' * 33)
                # print(self.objects[ex_index])
                # print(position_lists[0])
                # print(position_lists[1])

            if len(position_lists) == 1:
                spatial_objects[ex_index, :, position_lists[0]] = self.objects[ex_index].transpose(0, 1)  #.unsqueeze(-1)

            elif len(position_lists) == 2:
                # try:
                spatial_objects[ex_index, :, position_lists[0],
                                position_lists[1]] = self.objects[ex_index].transpose(0, 1)  #.unsqueeze(-1)
                # except IndexError as e:
                #     print('OBJECTS:')
                #     print(self.objects[ex_index])
                #     print('POSITION LISTS:')
                #     print(position_lists[0])
                #     print(position_lists[1])
                #     print('VALUES:')
                #     print(self.objects[ex_index].transpose(0, 1))
                #     print('MAXES:')
                #     print([x.max() for x in (position_lists[0], position_lists[1], self.objects[ex_index])])
                #     raise e

            elif len(position_lists) == 3:
                spatial_objects[ex_index, :, position_lists[0], position_lists[1],
                                position_lists[2]] = self.objects[ex_index].transpose(0, 1)  #.unsqueeze(-1)

        self.spatial_objects = spatial_objects

    def get_object_size(self):
        return self.spatial_objects.shape[1]

    def __getitem__(self, item):
        return self.spatial_objects[item], self.labels[item]

    def __len__(self):
        return self.spatial_objects.shape[0]


class SimplifiedSpatialDataset(MinimalSpatialDataset):
    def __init__(self, objects, labels, object_generator, x_max, y_max, position_indices=(0, 1)):
        super(SimplifiedSpatialDataset, self).__init__(objects, labels, object_generator,
                                                       x_max, y_max, position_indices)
        self.spatial_objects = self.spatial_objects[:, self.object_generator.get_type_slice(), :, :]


class QuinnBaseDatasetGenerator:
    def __init__(self, object_generator, x_max, y_max, seed, *,
        spatial_dataset=False, prop_train_to_validation=0.1, subsample_train_size=None):

        self.object_generator = object_generator
        self.x_max = x_max
        self.y_max = y_max
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.spatial_dataset = spatial_dataset
        self.prop_train_to_validation = prop_train_to_validation
        self.subsample_train_size = subsample_train_size

        self.train_dataset = None
        self.validation_dataset = None
        self.test_datasets = None


    @abstractmethod
    def _create_training_dataset(self) -> torch.utils.data.Dataset:
        raise NotImplementedError()

    @abstractmethod
    def _create_test_datasets(self) -> dict:
        raise NotImplementedError()

    def get_training_dataset(self) -> torch.utils.data.Dataset:
        if self.train_dataset is None:
            self.train_dataset = self._create_training_dataset()

            if self.prop_train_to_validation is not None and self.prop_train_to_validation > 0:
                train, val = self._split_dataset(self.train_dataset, 1 - self.prop_train_to_validation)
                self.train_dataset = train
                self.validation_dataset = val

            if self.subsample_train_size is not None:
                self.train_dataset.subsample(self.rng, total=self.subsample_train_size)

        return self.train_dataset

    def get_validation_dataset(self) -> torch.utils.data.Dataset:
        if self.validation_dataset is not None:
            return self.validation_dataset

        if self.prop_train_to_validation is not None and self.prop_train_to_validation > 0:
            self.get_training_dataset()

        return self.validation_dataset

    def get_test_datasets(self) -> dict:
        if self.test_datasets is None:
            self.test_datasets = self._create_test_datasets()

        return self.test_datasets

    def create_input(self, target, *references, train=True) -> torch.Tensor:
        return torch.cat([self.object_generator.target_object(target[0], target[1], train)] +
                         [self.object_generator.reference_object(reference[0], reference[1], train)
                          for reference in references])

    def _create_dataset(self, objects, labels):
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)

        if self.spatial_dataset:
            spatial_dataset_class = MinimalSpatialDataset
            if isinstance(self.spatial_dataset, str) and self.spatial_dataset.lower() == 'simplified':
                spatial_dataset_class = SimplifiedSpatialDataset

            return spatial_dataset_class(objects, labels, self.object_generator, self.x_max, self.y_max)

        return MinimalDataset(objects, labels, self.object_generator)

    def _split_train_test(self, items, prop_train=None, max_train_index=None):
        if prop_train is None and max_train_index is None:
            raise ValueError('Must provide _split_train_test with either prop_train or max_train_index')

        self.rng.shuffle(items)

        if max_train_index is None:
            max_train_index = int(np.floor(len(items) * prop_train))
        return items[:max_train_index], items[max_train_index:]

    def _split_dataset(self, dataset, prop_split=None, split_index=None):
        if prop_split is None and split_index is None:
            raise ValueError('Must provide _split_dataset with either prop_split or split_index')

        perm = self.rng.permutation(np.arange(len(dataset)))

        if split_index is None:
            split_index = int(np.floor(len(dataset) * prop_split))

        first_split = perm[:split_index]
        second_split = perm[split_index:]

        return (self._create_dataset(dataset.objects[first_split], dataset.labels[first_split]),
                self._create_dataset(dataset.objects[second_split], dataset.labels[second_split]))

        
class QuinnWithReferenceDatasetGenerator(QuinnBaseDatasetGenerator):
    def __init__(self, object_generator, x_max, y_max, seed, *,
                 prop_train_reference_object_locations=0.8, prop_train_target_object_locations=0.5,
                 reference_object_x_margin=0, reference_object_y_margin_bottom=0, reference_object_y_margin_top=0,
                 spatial_dataset=False, prop_train_to_validation=0.1, subsample_train_size=None):
        
        super(QuinnWithReferenceDatasetGenerator, self).__init__(object_generator, x_max, y_max, seed,
            spatial_dataset=spatial_dataset, prop_train_to_validation=prop_train_to_validation, 
            subsample_train_size=subsample_train_size)

        self.prop_train_reference_object_locations = prop_train_reference_object_locations
        self.prop_train_target_object_locations = prop_train_target_object_locations

        if reference_object_x_margin is None:
            reference_object_x_margin = 0

        self.reference_object_x_margin = reference_object_x_margin
        self.reference_object_y_margin_bottom = reference_object_y_margin_bottom
        self.reference_object_y_margin_top = reference_object_y_margin_top

        possible_reference_object_locations = [np.array(x) for x in
                                               itertools.product(range(reference_object_x_margin,
                                                                       x_max - reference_object_x_margin - object_generator.reference_object_length),
                                                                 range(reference_object_y_margin_bottom,
                                                                       y_max - reference_object_y_margin_top))]
        self.train_reference_object_locations, self.test_reference_object_locations = \
            self._split_train_test(possible_reference_object_locations, prop_train_reference_object_locations)


        self.train_target_locations, self.test_target_locations = self._generate_and_split_target_object_locations()

    @abstractmethod
    def _create_single_dataset(self, reference_locations, target_locations, train=True):
        raise NotImplementedError()

    @abstractmethod
    def _generate_and_split_target_object_locations(self, prop_train=None) -> list:
        raise NotImplementedError()

    def _create_training_dataset(self) -> torch.utils.data.Dataset:
        return self._create_single_dataset(self.train_reference_object_locations,
                                           self.train_target_locations, train=True)

    def _create_test_datasets(self) -> dict:
        test_datasets = dict()

        test_datasets[TRAIN_REFERENCE_TEST_TARGET] = self._create_single_dataset(
                self.train_reference_object_locations, self.test_target_locations, train=False)

        test_datasets[TEST_REFERENCE_TRAIN_TARGET] = self._create_single_dataset(
                self.test_reference_object_locations, self.train_target_locations, train=False)

        test_datasets[TEST_REFERENCE_TEST_TARGET] = self._create_single_dataset(
                self.test_reference_object_locations, self.test_target_locations, train=False)

        return test_datasets


TRAIN_REFERENCE_TEST_TARGET = 'train_reference_test_target'
TEST_REFERENCE_TRAIN_TARGET = 'test_reference_train_target'
TEST_REFERENCE_TEST_TARGET = 'test_reference_test_target'


class CombinedQuinnDatasetGenerator(QuinnWithReferenceDatasetGenerator):
    def __init__(self, object_generator, x_max, y_max, seed, *,
                 between_relation=False, two_reference_objects=None,
                 prop_train_target_object_locations=0.5,
                 prop_train_reference_object_locations=0.8,
                 target_object_grid_height=8, adjacent_reference_objects=False,
                 reference_object_x_margin=0,
                 reference_object_y_margin_bottom=None, reference_object_y_margin_top=None,
                 spatial_dataset=False,
                 prop_train_to_validation=0.1, subsample_train_size=None):

        self.between_relation = between_relation
        if two_reference_objects is None:
            two_reference_objects = between_relation
        self.two_reference_objects = two_reference_objects

        if target_object_grid_height % 4 != 0:
            raise ValueError(f'Target object grid height must be divisible by 4, received target_object_grid_height={target_object_grid_height}')

        if reference_object_y_margin_bottom is None:
            reference_object_y_margin_bottom = 0

        if reference_object_y_margin_top is None or reference_object_y_margin_top < target_object_grid_height:
            reference_object_y_margin_top = target_object_grid_height + 1 + int(self.two_reference_objects)

        
        self.target_object_grid_height = target_object_grid_height
        self.adjacent_reference_objects = adjacent_reference_objects
        
        self.single_reference_height = self.target_object_grid_height // 2

        if self.adjacent_reference_objects:
            if between_relation or not two_reference_objects:
                raise ValueError(f'adjacent_reference_objects=True requires between_relation=False and two_reference_objects=True')
            self.bottom_reference_height = self.single_reference_height
            self.top_reference_height = self.single_reference_height

        else:
            self.bottom_reference_height = self.target_object_grid_height // 4
            self.top_reference_height = self.target_object_grid_height * 3 // 4

        super(CombinedQuinnDatasetGenerator, self).__init__(
            object_generator=object_generator, x_max=x_max, y_max=y_max, seed=seed,
            prop_train_reference_object_locations=prop_train_reference_object_locations,
            prop_train_target_object_locations=prop_train_target_object_locations,
            reference_object_x_margin=reference_object_x_margin,
            reference_object_y_margin_bottom=reference_object_y_margin_bottom,
            reference_object_y_margin_top=reference_object_y_margin_top,
            spatial_dataset=spatial_dataset, prop_train_to_validation=prop_train_to_validation,
            subsample_train_size=subsample_train_size
        )

    def _generate_and_split_target_object_locations(self, prop_train=None):
        if prop_train is None:
            prop_train = self.prop_train_target_object_locations

        x_range = np.arange(self.object_generator.reference_object_length)

        if self.between_relation:
            y_ranges = (np.arange(self.bottom_reference_height),
                        np.arange(self.bottom_reference_height, self.top_reference_height),
                        np.arange(self.top_reference_height, self.target_object_grid_height))
        elif self.two_reference_objects and not self.adjacent_reference_objects:
            y_ranges = (np.arange(self.bottom_reference_height),
                        np.arange(self.top_reference_height, self.target_object_grid_height))
        else:
            y_ranges = (np.arange(self.single_reference_height),
                        np.arange(self.single_reference_height, self.target_object_grid_height))

        all_locations = [[np.array(x) for x in itertools.product(x_range, y_range)]
                         for y_range in y_ranges]
        split_locations = [self._split_train_test(locations, prop_train) for locations in all_locations]
        return [sum(split, list()) for split in zip(*split_locations)]

    def _create_single_dataset(self, reference_locations, target_locations, train=True):
        objects = []
        labels = []

        if self.between_relation:
            for grid_bottom_left_corner in reference_locations:
                bottom_reference_location = grid_bottom_left_corner + np.array([0, self.bottom_reference_height])
                # the + 1 accounts for the bottom object
                top_reference_location = grid_bottom_left_corner + np.array([0, self.top_reference_height + 1])

                for rel_target_location in target_locations:
                    target_location = grid_bottom_left_corner + rel_target_location
                    label = 0
                    if rel_target_location[1] >= self.top_reference_height:  # above both
                        target_location += np.array([0, 2])
                    elif rel_target_location[1] >= self.bottom_reference_height:  # above bottom only
                        target_location += np.array([0, 1])
                        label = 1

                    objects.append(self.create_input(target_location, bottom_reference_location,
                                                     top_reference_location, train=train))
                    labels.append(label)

        else:  # above/below
            for grid_bottom_left_corner in reference_locations:
                reference_location = grid_bottom_left_corner + np.array([0, self.single_reference_height])
                bottom_reference_location = grid_bottom_left_corner + np.array([0, self.bottom_reference_height])
                # the + 1 accounts for the bottom object
                top_reference_location = grid_bottom_left_corner + np.array([0, self.top_reference_height + 1])

                for rel_target_location in target_locations:
                    target_location = grid_bottom_left_corner + rel_target_location
                    label = 0
                    if rel_target_location[1] >= self.single_reference_height:  # above
                        target_location += np.array([0, 1 + int(self.two_reference_objects)])  
                        label = 1

                    if self.two_reference_objects:
                        objects.append(self.create_input(target_location, bottom_reference_location, top_reference_location, train=train))
                    else:
                        objects.append(self.create_input(target_location, reference_location, train=train))
                    labels.append(label)

        return self._create_dataset(objects, labels)

class DiagonalAboveBelowDatasetGenerator(QuinnWithReferenceDatasetGenerator):
    def __init__(self, object_generator, x_max, y_max, seed, *,
                 reference_object_x_margin=0, reference_object_y_margin=0,
                 prop_train_reference_object_locations=0.8, prop_train_target_object_locations=0.5,
                 spatial_dataset=False, prop_train_to_validation=0.1, subsample_train_size=None):

        if not isinstance(object_generator, DiagonalObjectGeneratorWithoutSize):
            raise ValueError(f'object_generator must be a DiagonalObjectGeneratorWithoutSize, got {type(object_generator)}')

        if reference_object_y_margin is None:
            reference_object_y_margin = 0

        self.reference_object_length = object_generator.reference_object_length
        reference_object_y_margin_bottom = reference_object_y_margin
        reference_object_y_margin_top = self.reference_object_length + reference_object_y_margin

        self.indices_above = None
        self.indices_below = None

        super(DiagonalAboveBelowDatasetGenerator, self).__init__(
            object_generator=object_generator, x_max=x_max, y_max=y_max, seed=seed,
            prop_train_reference_object_locations=prop_train_reference_object_locations,
            prop_train_target_object_locations=prop_train_target_object_locations,
            reference_object_x_margin=reference_object_x_margin,
            reference_object_y_margin_bottom=reference_object_y_margin_bottom,
            reference_object_y_margin_top=reference_object_y_margin_top,
            spatial_dataset=spatial_dataset, prop_train_to_validation=prop_train_to_validation,
            subsample_train_size=subsample_train_size
        )

    def _generate_and_split_target_object_locations(self, prop_train=None) -> list:
        if prop_train is None:
            prop_train = self.prop_train_target_object_locations

        self.indices_above = list(zip(*np.triu_indices(self.reference_object_length, 1)))
        self.indices_below = list(zip(*np.tril_indices(self.reference_object_length, -1)))

        split_locations = [self._split_train_test(locations, prop_train) 
            for locations in (self.indices_above, self.indices_below)]

        return [sum(split, list()) for split in zip(*split_locations)]

    def _create_single_dataset(self, reference_locations, target_locations, train=True):
        objects = []
        labels = []

        for grid_upper_right_corner in reference_locations:
            reference_start_location = grid_upper_right_corner

            for rel_target_location in target_locations:
                label = int(rel_target_location[0] < rel_target_location[1])
                target_location = grid_upper_right_corner + rel_target_location

                objects.append(self.create_input(target_location, reference_start_location, train=train))
                labels.append(label)

        return self._create_dataset(objects, labels)

class QuinnNoReferenceDatasetGenerator(QuinnBaseDatasetGenerator):
    def __init__(self, object_generator, x_max, y_max, seed, *,
        left_right=True, prop_train=0.9,
        spatial_dataset=False, prop_train_to_validation=0.1, subsample_train_size=None):

        if x_max % 2 != 0 or y_max % 2 != 0:
            raise ValueError(f'x_max and y_max must be even, got {x_max} and {y_max}')

        super(QuinnNoReferenceDatasetGenerator, self).__init__(object_generator, x_max, y_max, seed,
            spatial_dataset=spatial_dataset, prop_train_to_validation=prop_train_to_validation, 
            subsample_train_size=subsample_train_size)

        self.left_right = left_right
        self.prop_train = prop_train

        all_target_locations = list(itertools.product(range(self.x_max), range(self.y_max)))
        self.label_to_locations = {}
        self.stage_to_label_to_locations = defaultdict(dict)

        compare_point = x_max // 2 if left_right else y_max // 2
        compare_index = 0 if left_right else 1
        
        self.label_to_locations[0] = [location for location in all_target_locations if location[compare_index] < compare_point]
        self.label_to_locations[1] = [location for location in all_target_locations if location[compare_index] >= compare_point]

        for label in 0, 1:
            for stage, locations in zip(('train', 'test'), self._split_train_test(self.label_to_locations[label], self.prop_train)):
                self.stage_to_label_to_locations[stage][label] = locations

    def _create_stage_dataset(self, stage):
        objects = []
        labels = []

        for label, locations in self.stage_to_label_to_locations[stage].items():
            for target in locations:
                objects.append(self.object_generator.target_object(target[0], target[1], train=stage == 'train'))
                labels.append(label)

        return self._create_dataset(objects, labels)

    def _create_training_dataset(self) -> torch.utils.data.Dataset:
        return self._create_stage_dataset('train')

    def _create_test_datasets(self) -> dict:
        return {'test': self._create_stage_dataset('test')}
