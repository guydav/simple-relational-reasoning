from collections import namedtuple
import numpy as np
import torch

from simple_relational_reasoning.datagen.object_fields import FIELD_TYPES


FieldConfig = namedtuple('FieldConfig', ('name', 'type', 'kwargs'))
FieldConfig.__new__.__defaults__ = (None, None, dict())


DEFAULT_POSITION_FIELDS = ('x', 'y')


def no_position_collision_constraint(object_batch, relevant_indices, field_slices,
                                     position_fields=DEFAULT_POSITION_FIELDS):
    violating_indices = []

    if relevant_indices is None:
        relevant_indices = range(object_batch.shape[0])

    for idx in relevant_indices:
        object_positions = torch.cat([object_batch[idx, :, field_slices[pos]] for pos in position_fields],
                                     dim=1).to(torch.float)

        for obj_idx in range(object_positions.shape[0] - 1):
            if (object_positions[obj_idx + 1:] == object_positions[obj_idx]).all(dim=1).any():
                violating_indices.append(idx)
                break

    return violating_indices


DEFAULT_CONSTRAINTS = (
    no_position_collision_constraint,
)


class ObjectGenerator:
    def __init__(self, n, field_configs, relation_class, constraints=DEFAULT_CONSTRAINTS,
                 batch_size=1, object_dtype=torch.float, label_dtype=torch.long, relation_kwargs=None):
        self.n = n
        self.field_configs = field_configs
        assert(all([cfg.type in FIELD_TYPES for cfg in field_configs]))

        if object_dtype is not None:
            for cfg in field_configs:
                cfg.kwargs['dtype'] = object_dtype

        self.field_generators = {cfg.name: FIELD_TYPES[cfg.type](n, **cfg.kwargs)
                                 for cfg in field_configs}
        cum_lengths = list(np.cumsum([len(self.field_generators[name]) for name in self.field_generators]))
        cum_lengths.insert(0, 0)
        slices = [slice(start, end) for start, end in zip(cum_lengths[:-1], cum_lengths[1:])]
        self.field_slices = {name: slices[i] for i, name in enumerate(self.field_generators)}

        if relation_kwargs is None:
            relation_kwargs = {}

        self.relation_class = relation_class
        self.relation = self.relation_class(self.field_slices, self.field_generators, **relation_kwargs)
        self.constraints = constraints
        self.batch_size = batch_size
        self.object_dtype = object_dtype

        if label_dtype is None:
            label_dtype = object_dtype
        self.label_dtype = label_dtype

        self.object_size = self(1)[0].shape[-1]

    def __call__(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        batch_tensor = torch.stack([torch.cat([gen() for gen in self.field_generators.values()], dim=1)
                                    for _ in range(batch_size)])

        if self.constraints is not None:
            any_violations = True
            prev_violating_indices = None

            while any_violations:
                violating_indices = set()

                for constraint in self.constraints:
                    violating_indices.update(constraint(batch_tensor, prev_violating_indices, self.field_slices))

                n_violations = len(violating_indices)

                if n_violations == 0:
                    any_violations = False

                else:

                    violating_indices = sorted(violating_indices)
                    new_elements = torch.stack([torch.cat([gen() for gen in self.field_generators.values()], dim=1)
                                    for _ in range(n_violations)])
                    batch_tensor[violating_indices] = new_elements
                    prev_violating_indices = violating_indices

        batch_labels = torch.tensor([self.relation.evaluate(batch_tensor[i])
                                     for i in range(batch_size)], dtype=self.label_dtype)
        return batch_tensor, batch_labels


class BalancedBatchObjectGenerator(ObjectGenerator):
    def __init__(self, n, field_configs, relation_class, batch_size=1, constraints=DEFAULT_CONSTRAINTS,
                 object_dtype=torch.float, label_dtype=torch.long, relation_kwargs=None):
        super(BalancedBatchObjectGenerator, self).__init__(
            n=n, field_configs=field_configs, relation_class=relation_class, constraints=constraints,
            batch_size=batch_size, object_dtype=object_dtype, label_dtype=label_dtype, relation_kwargs=relation_kwargs)

    def __call__(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if batch_size == 1:
            return super(BalancedBatchObjectGenerator, self).__call__(1)

        half_batch_size = int(batch_size / 2)
        negative_examples, positive_examples = [], []
        negative_count, positive_count = 0, 0
        while negative_count < half_batch_size or positive_count < half_batch_size:
            data, labels = super(BalancedBatchObjectGenerator, self).__call__(batch_size)
            positive_locations = labels.bool()
            batch_positive_count = labels.sum()

            if positive_count < half_batch_size:
                positive_examples.append(data[positive_locations])
                positive_count += batch_positive_count

            if negative_count < half_batch_size:
                negative_examples.append(data[~positive_locations])
                negative_count += batch_size - batch_positive_count

        data = torch.cat((torch.cat(positive_examples)[:half_batch_size],
                           torch.cat(negative_examples)[:half_batch_size]))
        labels = torch.cat((torch.ones(half_batch_size, dtype=self.label_dtype),
                            torch.zeros(half_batch_size, dtype=self.label_dtype)))
        perm = torch.randperm(batch_size)
        return data[perm], labels[perm]


class SmartBalancedBatchObjectGenerator(ObjectGenerator):
    def __init__(self, n, field_configs, relation_class, negative_to_positive=True, constraints=DEFAULT_CONSTRAINTS,
                 batch_size=1, object_dtype=torch.float, label_dtype=torch.long, relation_kwargs=None,
                 max_recursion_depth=20):
        super(SmartBalancedBatchObjectGenerator, self).__init__(
            n=n, field_configs=field_configs, relation_class=relation_class, constraints=constraints,
            batch_size=batch_size, object_dtype=object_dtype, label_dtype=label_dtype, relation_kwargs=relation_kwargs)
        self.negative_to_positive = negative_to_positive
        self.max_recursion_depth = max_recursion_depth

    def __call__(self, batch_size=None, recursion_depth=0):
        if batch_size is None:
            batch_size = self.batch_size

        if batch_size == 1:
            return super(SmartBalancedBatchObjectGenerator, self).__call__(1)

        if recursion_depth > self.max_recursion_depth:
            raise ValueError('Object generator max recursion depth exceeded...')

        data, labels = super(SmartBalancedBatchObjectGenerator, self).__call__(batch_size)
        positive_count = int(labels.sum())
        half_batch_size = int(batch_size / 2)
        if positive_count == half_batch_size:
            return data, labels

        # TODO: eventually one could assume having strategies to balance in both directions. Account for that.

        more_positive_examples = positive_count > half_batch_size
        # either more positive examples and converting positives to negatives or vice versa
        while more_positive_examples == self.negative_to_positive:
            recursion_depth += 1
            if recursion_depth > self.max_recursion_depth:
                raise ValueError('Object generator max recursion depth exceeded...')

            indices_to_resample = torch.nonzero(labels.bool() == self.negative_to_positive).squeeze()
            num_to_resample = len(indices_to_resample)
            resampled_data, resampled_labels = super(SmartBalancedBatchObjectGenerator, self).__call__(num_to_resample)

            data[indices_to_resample] = resampled_data
            labels[indices_to_resample] = resampled_labels

            positive_count = int(labels.sum())
            if positive_count == half_batch_size:
                return data, labels

            more_positive_examples = positive_count > half_batch_size

        # At this point, we're guaranteed to be able to modify in the direction the balancer supports
        indices_to_modify = torch.nonzero(labels.bool() == (not self.negative_to_positive)).squeeze()
        num_samples_to_modify = abs(positive_count - half_batch_size)
        target_indices = indices_to_modify[torch.randperm(indices_to_modify.shape[0])[:num_samples_to_modify]]
        for index in target_indices:
            data[index] = self.relation.balance(data[index], not self.negative_to_positive)
            labels[index] = self.negative_to_positive

        return data, labels


class ObjectGeneratorDataset(torch.utils.data.Dataset):
    def __init__(self, object_generator, epoch_size):
        super(ObjectGeneratorDataset, self).__init__()
        self.object_generator = object_generator
        self.epoch_size = epoch_size
        self.objects = None
        self.labels = None
        self.regenerate()

    def regenerate(self):
        if self.epoch_size > 0:
            self.objects, self.labels = self.object_generator(self.epoch_size)

    def __getitem__(self, item):
        return self.objects[item], self.labels[item]

    def __len__(self):
        return self.epoch_size


class SpatialObjectGeneratorDataset(ObjectGeneratorDataset):
    def __init__(self, object_generator, epoch_size, position_fields=DEFAULT_POSITION_FIELDS):
        self.position_fields = position_fields
        self.position_field_generators = [object_generator.field_generators[p] for p in self.position_fields]
        super(SpatialObjectGeneratorDataset, self).__init__(object_generator=object_generator, epoch_size=epoch_size)

    def regenerate(self):
        super(SpatialObjectGeneratorDataset, self).regenerate()
        self.convert_objects()

    def convert_objects(self):
        if self.objects is None or len(self.objects.shape) == 2 or self.objects.shape[0] == 0:
            return

        D, N, O = self.objects.shape

        position_shape = [field.max_coord - field.min_coord for field in self.position_field_generators]
        spatial_shape = (D, O, *position_shape)
        spatial_objects = torch.zeros(spatial_shape, dtype=self.objects.dtype)
        for ex_index in range(D):
            # TODO: if this work, could probably flatten it again, but I don't think it's worth optimizing
            position_lists = [self.objects[ex_index, :, self.object_generator.field_slices[name]].long()
                              for name in self.position_fields]

            if len(position_lists) == 1:
                spatial_objects[ex_index, :, position_lists[0]] = self.objects[ex_index].transpose(0, 1).unsqueeze(-1)

            elif len(position_lists) == 2:
                spatial_objects[ex_index, :, position_lists[0],
                position_lists[1]] = self.objects[ex_index].transpose(0, 1).unsqueeze(-1)

            elif len(position_lists) == 3:
                spatial_objects[ex_index, :, position_lists[0],
                position_lists[1], position_lists[2]] = self.objects[ex_index].transpose(0, 1).unsqueeze(-1)

            # for obj_index in range(N):
            #     object_position = [self.objects[ex_index, obj_index, self.object_generator.field_slices[name]]
            #                        for name in self.position_fields]
            #     if len(object_position) == 1:
            #         spatial_objects[ex_index, object_position[0]] = self.objects[ex_index, obj_index]
            #
            #     elif len(object_position) == 2:
            #         spatial_objects[ex_index, object_position[0],
            #                         object_position[1]] = self.objects[ex_index, obj_index]
            #
            #     elif len(object_position) == 3:
            #         spatial_objects[ex_index, object_position[0],
            #                         object_position[1], object_position[2]] = self.objects[ex_index, obj_index]
            #
            #     else:
            #         raise ValueError(f'Currently only accounting for up to 3D positions, got {len(object_position)} dimensions: {object_position}')
        self.objects = spatial_objects


class ObjectGeneratorIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, object_generator, epoch_size):
        super(ObjectGeneratorIterableDataset, self).__init__()
        self.object_generator = object_generator
        self.epoch_size = epoch_size

    def __iter__(self):
        objects, labels = self.object_generator(self.epoch_size)

        def generator():
            for i in range(self.epoch_size):
                yield objects[i], labels[i]

        return generator()
