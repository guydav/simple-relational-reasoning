from collections import namedtuple
import numpy as np
import torch

from simple_relational_reasoning.datagen.object_fields import FIELD_TYPES


FieldConfig = namedtuple('FieldConfig', ('name', 'type', 'params'))
FieldConfig.__new__.__defaults__ = (None,) * len(FieldConfig._fields)


class ObjectGenerator:
    def __init__(self, n, field_configs, relation_evaluator, dtype=torch.int, label_dtype=torch.bool):
        self.n = n
        self.field_configs = field_configs
        assert(all([cfg.type in FIELD_TYPES for cfg in field_configs]))
        self.field_generators = {cfg.name: FIELD_TYPES[cfg.type](n, *cfg.params)
                                 for cfg in field_configs}
        cum_lengths = list(np.cumsum([len(self.field_generators[name]) for name in self.field_generators]))
        cum_lengths.insert(0, 0)
        slices = [slice(start, end) for start, end in zip(cum_lengths[:-1], cum_lengths[1:])]
        self.field_slices = {name: slices[i] for i, name in enumerate(self.field_generators)}

        self.relation_evaluator = relation_evaluator
        self.dtype = dtype
        self.label_dtype = label_dtype

    def __call__(self, batch_size=1):
        batch_tensor = torch.stack([torch.cat([gen() for gen in self.field_generators.values()], dim=1)
                                    for _ in range(batch_size)])
        batch_labels = torch.tensor([self.relation_evaluator(batch_tensor[i], self.field_slices)
                                     for i in range(batch_size)], dtype=self.label_dtype)
        return batch_tensor, batch_labels


# Implementing a quick example one, this one checks that two fields named 'x' and 'y' differ by some minimal amount
def adjacent_relation_evaluator(objects, field_slices, x_field_name='x', y_field_name='y'):
    assert(x_field_name in field_slices)
    assert(y_field_name in field_slices)

    object_positions = torch.cat((objects[:, field_slices[x_field_name]], objects[:, field_slices[y_field_name]]), dim=1).to(torch.float).unsqueeze(0)
    l1_distances = torch.cdist(object_positions, object_positions, 1)
    return torch.any(torch.isclose(l1_distances, torch.tensor([1.0])))

