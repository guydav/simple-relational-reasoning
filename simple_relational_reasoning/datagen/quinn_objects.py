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
                 n_train_target_types=1, n_test_target_types=0, dtype=torch.float):
        self.seed = seed
        self.reference_object_length = reference_object_length
        self.target_object_length = target_object_length
        self.n_reference_types = n_reference_types
        self.n_train_target_types = n_train_target_types
        self.n_test_target_types = n_test_target_types
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

        return self.rng.integers(self.n_reference_types + self.n_train_target_types,
                                 self.n_reference_types + self.n_train_target_types + self.n_test_target_types)

    def _to_one_hot(self, n, n_types):
        one_hot = np.zeros(n_types)
        one_hot[n] = 1
        return one_hot

    def _sample_type_one_hot(self, target=False, train=True):
        return self._to_one_hot(self._sample_type(target, train), self.n_reference_types + self.n_train_target_types)


class ObjectGeneratorWithSize(ObjectGenerator):
    def __init__(self, seed, reference_object_length, target_object_length=1, n_reference_types=1,
                 n_train_target_types=1, n_test_target_types=0):
        super(ObjectGeneratorWithSize, self).__init__(seed, reference_object_length, target_object_length,
                                                      n_reference_types, n_train_target_types, n_test_target_types)

    def reference_object(self, x, y, train=True):
        return torch.tensor([x, y, self.reference_object_length, *self._sample_type_one_hot(False, train)],
                            dtype=self.dtype).unsqueeze(0)

    def target_object(self, x, y, train=True):
        return torch.tensor([x, y, self.target_object_length, *self._sample_type_one_hot(True, train)],
                            dtype=self.dtype).unsqueeze(0)


class ObjectGeneratorWithoutSize(ObjectGenerator):
    def __init__(self, seed, reference_object_length, target_object_length=1, n_reference_types=1,
                 n_train_target_types=1, n_test_target_types=0):
        super(ObjectGeneratorWithoutSize, self).__init__(seed, reference_object_length, target_object_length,
                                                         n_reference_types, n_train_target_types, n_test_target_types)

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


class MinimalDataset(torch.utils.data.Dataset):
    def __init__(self, objects, labels):
        super(MinimalDataset, self).__init__()

        if not isinstance(objects, torch.Tensor):
            objects = torch.stack(objects)

        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)

        self.objects = objects
        self.labels = labels

    def __getitem__(self, item):
        return self.objects[item], self.labels[item]

    def __len__(self):
        return self.objects.shape[0]


class MinimalSpatialDataset(MinimalDataset):
    def __init__(self, objects, labels, x_max, y_max, position_indices=(0, 1)):
        super(MinimalSpatialDataset, self).__init__(objects, labels)

        D, N, O = self.objects.shape

        position_shape = (x_max, y_max)
        spatial_shape = (D, O, *position_shape)
        spatial_objects = torch.zeros(spatial_shape, dtype=self.objects.dtype)
        for ex_index in range(D):
            position_lists = [self.objects[ex_index, :, index].long()
                              for index in position_indices]

            if len(position_lists) == 1:
                spatial_objects[ex_index, :, position_lists[0]] = self.objects[ex_index].transpose(0, 1)  #.unsqueeze(-1)

            elif len(position_lists) == 2:
                spatial_objects[ex_index, :, position_lists[0],
                                position_lists[1]] = self.objects[ex_index].transpose(0, 1)  #.unsqueeze(-1)

            elif len(position_lists) == 3:
                spatial_objects[ex_index, :, position_lists[0], position_lists[1],
                                position_lists[2]] = self.objects[ex_index].transpose(0, 1)  #.unsqueeze(-1)

        self.spatial_objects = spatial_objects

    def __getitem__(self, item):
        return self.spatial_objects[item], self.labels[item]

    def __len__(self):
        return self.spatial_objects.shape[0]


class QuinnDatasetGenerator:
    def __init__(self, object_generator, x_max, y_max, seed, *,
                 add_neither_train=True, add_neither_test=False, prop_train_reference_object_locations=0.8,
                 reference_object_x_margin=0, reference_object_y_margin_bottom=0, reference_object_y_margin_top=0,
                 spatial_dataset=False):
        self.object_generator = object_generator
        self.x_max = x_max
        self.y_max = y_max
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.add_neither_train = add_neither_train
        self.add_neither_test = add_neither_test

        self.prop_train_reference_object_locations = prop_train_reference_object_locations
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

        self.spatial_dataset = spatial_dataset
        self.train_dataset = None
        self.test_datasets = None

    def get_training_dataset(self) -> torch.utils.data.Dataset:
        raise NotImplementedError()

    def get_test_datasets(self) -> dict:
        raise NotImplementedError()

    def create_input(self, target, *references, train=True) -> torch.Tensor:
        return torch.cat([self.object_generator.target_object(target[0], target[1], train)] +
                         [self.object_generator.reference_object(reference[0], reference[1], train)
                          for reference in references])

    def _split_train_test(self, items, prop_train=None, max_train_index=None):
        self.rng.shuffle(items)

        if prop_train is None and max_train_index is None:
            raise ValueError('Must provide _split_train_test with either prop_train or max_train_index')

        if max_train_index is None:
            max_train_index = int(len(items) * prop_train)
        return items[:max_train_index], items[max_train_index:]

    def _create_dataset(self, objects, labels):
        if self.spatial_dataset:
            return MinimalSpatialDataset(objects, labels, self.x_max, self.y_max)

        return MinimalDataset(objects, labels)


TRAIN_REFERENCE_TEST_TARGET = 'train_reference_test_target'
TRAIN_REFERENCE_MIDDLE_TARGET = 'train_reference_middle_target'
TEST_REFERENCE_TRAIN_TARGET = 'test_reference_train_target'
TEST_REFERENCE_TEST_TARGET = 'test_reference_test_target'
TEST_REFERENCE_MIDDLE_TARGET = 'test_reference_middle_target'


class ReferenceInductiveBias(QuinnDatasetGenerator):
    def __init__(self, object_generator, x_max, y_max, seed, *,
                 target_object_grid_size=3, add_neither_train=True, above_or_between_left=None,
                 n_train_target_object_locations=None, prop_train_reference_object_locations=0.8,
                 reference_object_x_margin=0, reference_object_y_margin_bottom=None,
                 reference_object_y_margin_top=None, add_neither_test=False, spatial_dataset=False):
        if reference_object_y_margin_bottom is None:
            reference_object_y_margin_bottom = target_object_grid_size

        if reference_object_y_margin_top is None:
            reference_object_y_margin_top = target_object_grid_size

        super(ReferenceInductiveBias, self).__init__(
            object_generator=object_generator, x_max=x_max, y_max=y_max, seed=seed,
            add_neither_train=add_neither_train, add_neither_test=add_neither_test,
            prop_train_reference_object_locations=prop_train_reference_object_locations,
            reference_object_x_margin=reference_object_x_margin,
            reference_object_y_margin_bottom=reference_object_y_margin_bottom,
            reference_object_y_margin_top=reference_object_y_margin_top, spatial_dataset=spatial_dataset
        )

        self.target_object_grid_size = target_object_grid_size

        if above_or_between_left is None:
            above_or_between_left = self.seed % 2
        self.above_or_between_left = above_or_between_left

        if n_train_target_object_locations is None:
            n_train_target_object_locations = target_object_grid_size * (target_object_grid_size - 1)
        self.n_train_target_object_locations = n_train_target_object_locations

        possible_target_object_locations = [np.array(x) for x in
                                            itertools.product(range(target_object_grid_size),
                                                              range(1, target_object_grid_size + 1))]

        self.train_target_object_locations, self.test_target_object_locations = \
            self._split_train_test(possible_target_object_locations, max_train_index=n_train_target_object_locations)

        self.middle_target_object_locations = [np.array(x) for x in
                                               itertools.product(range(target_object_grid_size, self.object_generator.reference_object_length - target_object_grid_size),
                                                                 range(1, target_object_grid_size + 1))]

    def _create_left_right_dataset(self, reference_locations, target_locations, train=True):
        raise NotImplementedError()

    def _create_middle_dataset(self, reference_locations, middle_locations=None):
        raise NotImplementedError()

    def get_training_dataset(self):
        if self.train_dataset is None:
            self.train_dataset = self._create_left_right_dataset(self.train_reference_object_locations,
                                                                 self.train_target_object_locations, train=True)

        return self.train_dataset

    def get_test_datasets(self):
        if self.test_datasets is None or not len(self.test_datasets):
            self.test_datasets = dict()

            self.test_datasets[TRAIN_REFERENCE_TEST_TARGET] = self._create_left_right_dataset(
                self.train_reference_object_locations, self.test_target_object_locations, train=False)

            self.test_datasets[TRAIN_REFERENCE_MIDDLE_TARGET] = self._create_middle_dataset(
                self.train_reference_object_locations, self.middle_target_object_locations)

            self.test_datasets[TEST_REFERENCE_TRAIN_TARGET] = self._create_left_right_dataset(
                self.test_reference_object_locations, self.train_target_object_locations, train=False)

            self.test_datasets[TEST_REFERENCE_TEST_TARGET] = self._create_left_right_dataset(
                self.test_reference_object_locations, self.test_target_object_locations, train=False)

            self.test_datasets[TEST_REFERENCE_MIDDLE_TARGET] = self._create_middle_dataset(
                self.test_reference_object_locations, self.middle_target_object_locations)

        return self.test_datasets


class AboveBelowReferenceInductiveBias(ReferenceInductiveBias):
    def __init__(self, object_generator, x_max, y_max, seed, *,
                 target_object_grid_size=3, add_neither_train=True, above_or_between_left=None,
                 n_train_target_object_locations=None, prop_train_reference_object_locations=0.8,
                 reference_object_x_margin=0, reference_object_y_margin_bottom=None,
                 reference_object_y_margin_top=None, add_neither_test=False, spatial_dataset=False):

        super(AboveBelowReferenceInductiveBias, self).__init__(
            object_generator=object_generator, x_max=x_max, y_max=y_max, seed=seed,
            target_object_grid_size=target_object_grid_size, add_neither_train=add_neither_train,
            n_train_target_object_locations=n_train_target_object_locations, above_or_between_left=above_or_between_left,
            prop_train_reference_object_locations=prop_train_reference_object_locations,
            reference_object_x_margin=reference_object_x_margin,
            reference_object_y_margin_bottom=reference_object_y_margin_bottom,
            reference_object_y_margin_top=reference_object_y_margin_top,
            add_neither_test=add_neither_test, spatial_dataset=spatial_dataset
        )

    def _create_left_right_dataset(self, reference_locations, target_locations, train=True):
        objects = []
        labels = []

        if train:
            add_neither = self.add_neither_train
        else:
            add_neither = self.add_neither_test

        for reference_location in reference_locations:
            reference_end = reference_location + np.array(
                [self.object_generator.reference_object_length - self.target_object_grid_size, 0])

            for target_location in target_locations:
                target_y_below = target_location + np.array([0, -4])
                if self.above_or_between_left:
                    objects.append(self.create_input(reference_location + target_location, reference_location, train=train))
                    objects.append(self.create_input(reference_end + target_y_below, reference_location, train=train))

                else:
                    objects.append(self.create_input(reference_end + target_location, reference_location, train=train))
                    objects.append(self.create_input(reference_location + target_y_below, reference_location, train=train))

                labels.extend([0 + int(add_neither), 1 + int(add_neither)])

            if add_neither:
                valid_x_locations = list(range(reference_location[0])) + list(
                    range(reference_location[0] + self.object_generator.reference_object_length, self.x_max))
                neither_x_locations = self.rng.choice(valid_x_locations, len(target_locations))
                neither_y_locations = self.rng.choice(range(self.y_max), len(target_locations))
                neither_locations = np.stack([neither_x_locations, neither_y_locations]).T
                for loc in neither_locations:
                    objects.append(self.create_input(loc, reference_location, train=train))
                    labels.append(0)

        return self._create_dataset(objects, labels)

    def _create_middle_dataset(self, reference_locations, middle_locations=None):
        objects = []
        labels = []

        if middle_locations is None:
            middle_locations = self.middle_target_object_locations

        for reference_location in reference_locations:
            for target_location in middle_locations:
                target_y_below = target_location + np.array([0, -4])
                objects.append(self.create_input(reference_location + target_location, reference_location, train=False))
                objects.append(self.create_input(reference_location + target_y_below, reference_location, train=False))
                labels.extend([0 + int(self.add_neither_test), 1 + int(self.add_neither_test)])

        return self._create_dataset(objects, labels)


class BetweenReferenceInductiveBias(ReferenceInductiveBias):
    def __init__(self, object_generator, x_max, y_max, seed, *,
                 target_object_grid_size=3, add_neither_train=True, above_or_between_left=None,
                 n_train_target_object_locations=None, prop_train_reference_object_locations=0.8,
                 reference_object_x_margin=0, reference_object_y_margin_bottom=None,
                 reference_object_y_margin_top=None, add_neither_test=False, spatial_dataset=False):

        # We assume that the generated reference object location is for the bottom reference object
        min_y_margin = 2 * target_object_grid_size + 1
        if reference_object_y_margin_top is None or reference_object_y_margin_top < min_y_margin:
            reference_object_y_margin_top = min_y_margin

        super(BetweenReferenceInductiveBias, self).__init__(
            object_generator=object_generator, x_max=x_max, y_max=y_max, seed=seed,
            target_object_grid_size=target_object_grid_size, add_neither_train=add_neither_train,
            n_train_target_object_locations=n_train_target_object_locations, above_or_between_left=above_or_between_left,
            prop_train_reference_object_locations=prop_train_reference_object_locations,
            reference_object_x_margin=reference_object_x_margin,
            reference_object_y_margin_bottom=reference_object_y_margin_bottom,
            reference_object_y_margin_top=reference_object_y_margin_top,
            add_neither_test=add_neither_test, spatial_dataset=spatial_dataset
        )

    def _create_left_right_dataset(self, reference_locations, target_locations, train=True):
        objects = []
        labels = []

        if train:
            add_neither = self.add_neither_train
        else:
            add_neither = self.add_neither_test

        for bottom_reference_location in reference_locations:
            bottom_reference_end = bottom_reference_location + np.array(
                [self.object_generator.reference_object_length - self.target_object_grid_size, 0])

            top_reference_location = bottom_reference_location + np.array(
                [0, self.target_object_grid_size + 1])

            top_reference_end = top_reference_location + np.array(
                [self.object_generator.reference_object_length - self.target_object_grid_size, 0])

            for target_location in target_locations:
                target_y_below = target_location + np.array([0, -4])
                if self.above_or_between_left:
                    objects.append(self.create_input(top_reference_end + target_location,
                                                     bottom_reference_location, top_reference_location, train=train))
                    objects.append(self.create_input(bottom_reference_end + target_y_below,
                                                     bottom_reference_location, top_reference_location, train=train))
                    objects.append(self.create_input(bottom_reference_location + target_location,
                                                     bottom_reference_location, top_reference_location, train=train))

                else:
                    objects.append(self.create_input(top_reference_location + target_location,
                                                     bottom_reference_location, top_reference_location, train=train))
                    objects.append(self.create_input(bottom_reference_location + target_y_below,
                                                     bottom_reference_location, top_reference_location, train=train))
                    objects.append(self.create_input(bottom_reference_end + target_location,
                                                     bottom_reference_location, top_reference_location, train=train))

                labels.extend([0 + int(add_neither), 1 + int(add_neither), 2 + int(add_neither)])

            if add_neither:
                valid_x_locations = list(range(bottom_reference_location[0])) + list(
                    range(bottom_reference_location[0] + self.object_generator.reference_object_length, self.x_max))
                neither_x_locations = self.rng.choice(valid_x_locations, len(target_locations))
                neither_y_locations = self.rng.choice(range(self.y_max), len(target_locations))
                neither_locations = np.stack([neither_x_locations, neither_y_locations]).T
                for loc in neither_locations:
                    objects.append(self.create_input(loc, bottom_reference_location, top_reference_location, train=train))
                    labels.append(0)

        return self._create_dataset(objects, labels)

    def _create_middle_dataset(self, reference_locations, middle_locations=None):
        objects = []
        labels = []

        if middle_locations is None:
            middle_locations = self.middle_target_object_locations

        for bottom_reference_location in reference_locations:
            top_reference_location = bottom_reference_location + np.array(
                [0, self.target_object_grid_size + 1])

            for target_location in middle_locations:
                target_y_below = target_location + np.array([0, -4])
                objects.append(self.create_input(top_reference_location + target_location,
                                                 bottom_reference_location, top_reference_location, train=False))
                objects.append(self.create_input(bottom_reference_location + target_y_below,
                                                 bottom_reference_location, top_reference_location, train=False))
                objects.append(self.create_input(bottom_reference_location + target_location,
                                                 bottom_reference_location, top_reference_location, train=False))
                labels.extend([0 + int(self.add_neither_test), 1 + int(self.add_neither_test), 
                               2 + int(self.add_neither_test)])

        return self._create_dataset(objects, labels)


class OneOrTwoReferenceObjects(QuinnDatasetGenerator):
    def __init__(self, object_generator, x_max, y_max, seed, *,
                 between_relation=False, two_reference_objects=None,
                 add_neither_train=True, prop_train_target_object_locations=0.5,
                 prop_train_reference_object_locations=0.8, reference_object_gap=3, reference_object_x_margin=0,
                 reference_object_y_margin_bottom=None, reference_object_y_margin_top=None, add_neither_test=False,
                 spatial_dataset=False):

        if reference_object_y_margin_bottom is None:
            reference_object_y_margin_bottom = reference_object_gap

        min_y_margin = 2 * reference_object_gap + 1
        if reference_object_y_margin_top is None or reference_object_y_margin_top < min_y_margin:
            reference_object_y_margin_top = min_y_margin

        super(OneOrTwoReferenceObjects, self).__init__(
            object_generator=object_generator, x_max=x_max, y_max=y_max, seed=seed,
            add_neither_train=add_neither_train, add_neither_test=add_neither_test,
            prop_train_reference_object_locations=prop_train_reference_object_locations,
            reference_object_x_margin=reference_object_x_margin,
            reference_object_y_margin_bottom=reference_object_y_margin_bottom,
            reference_object_y_margin_top=reference_object_y_margin_top,
            spatial_dataset=spatial_dataset
        )

        self.between_relation = between_relation
        if two_reference_objects is None:
            two_reference_objects = between_relation

        self.two_reference_objects = two_reference_objects

        self.reference_object_gap = reference_object_gap
        self.prop_train_target_object_locations = prop_train_target_object_locations

        self.train_below_target_locations, self.test_below_target_locations = \
            self._generate_and_split_target_object_locations(above=False)

        self.train_between_target_locations, self.test_between_target_locations = \
            self._generate_and_split_target_object_locations()

        self.train_above_target_locations, self.test_above_target_locations = \
            self._generate_and_split_target_object_locations()

    def _generate_and_split_target_object_locations(self, above=True, prop=None, x_range=None):
        if prop is None:
            prop = self.prop_train_target_object_locations

        if x_range is None:
            x_range = np.arange(self.object_generator.reference_object_length)

        y_range = np.arange(1, self.reference_object_gap + 1)
        if not above:
            y_range *= -1

        locations = [np.array(x) for x in itertools.product(x_range, y_range)]
        return self._split_train_test(locations, prop)

    def _create_single_dataset(self, reference_locations, targets_below,
                               targets_between, targets_above, train=True):
        objects = []
        labels = []
        
        if train:
            add_neither = self.add_neither_train
        else:
            add_neither = self.add_neither_test

        for bottom_reference_location in reference_locations:
            top_reference_location = bottom_reference_location + np.array([0, self.reference_object_gap + 1])

            for rel_target_location in targets_below:
                target_location = bottom_reference_location + rel_target_location
                if self.two_reference_objects:
                    objects.append(self.create_input(target_location, bottom_reference_location,
                                                     top_reference_location, train=train))
                else:
                    objects.append(self.create_input(target_location, bottom_reference_location, train=train))

                labels.append(0 + int(add_neither))

            for rel_target_location in targets_between:
                target_location = bottom_reference_location + rel_target_location
                if self.two_reference_objects:
                    objects.append(self.create_input(target_location, bottom_reference_location,
                                                     top_reference_location, train=train))
                else:
                    objects.append(self.create_input(target_location, bottom_reference_location, train=train))

                labels.append(1 + int(add_neither))

            for rel_target_location in targets_above:
                target_location = top_reference_location + rel_target_location
                if self.two_reference_objects:
                    objects.append(
                        self.create_input(target_location, bottom_reference_location,
                                          top_reference_location, train=train))
                else:
                    objects.append(self.create_input(target_location, bottom_reference_location, train=train))

                if self.between_relation:
                    labels.append(0 + int(add_neither))
                else:
                    labels.append(1 + int(add_neither))

            if add_neither:
                total_target_locations = (len(targets_below) + len(targets_between) + len(targets_above)) // 2

                valid_x_locations = list(range(bottom_reference_location[0])) + list(
                    range(bottom_reference_location[0] + self.object_generator.reference_object_length, self.x_max))
                neither_x_locations = self.rng.choice(valid_x_locations, total_target_locations)
                neither_y_locations = self.rng.choice(range(self.y_max), total_target_locations)
                neither_locations = np.stack([neither_x_locations, neither_y_locations]).T

                for loc in neither_locations:
                    if self.two_reference_objects:
                        objects.append(
                            self.create_input(loc, bottom_reference_location,
                                              top_reference_location, train=train))
                    else:
                        objects.append(self.create_input(loc, bottom_reference_location, train=train))

                    labels.append(0)

        return self._create_dataset(objects, labels)

    def get_training_dataset(self):
        if self.train_dataset is None:
            self.train_dataset = self._create_single_dataset(self.train_reference_object_locations,
                                                             self.train_below_target_locations,
                                                             self.train_between_target_locations,
                                                             self.train_above_target_locations, train=True)

        return self.train_dataset

    def get_test_datasets(self):
        if self.test_datasets is None or not len(self.test_datasets):
            self.test_datasets = dict()

            self.test_datasets[TRAIN_REFERENCE_TEST_TARGET] = self._create_single_dataset(
                self.train_reference_object_locations, self.test_below_target_locations,
                self.test_between_target_locations, self.test_above_target_locations, train=False)

            self.test_datasets[TEST_REFERENCE_TRAIN_TARGET] = self._create_single_dataset(
                self.test_reference_object_locations, self.train_below_target_locations,
                self.train_between_target_locations, self.train_above_target_locations, train=False)

            self.test_datasets[TEST_REFERENCE_TEST_TARGET] = self._create_single_dataset(
                self.test_reference_object_locations, self.test_below_target_locations,
                self.test_between_target_locations, self.test_above_target_locations, train=False)

        return self.test_datasets
