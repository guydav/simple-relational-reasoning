import torch


class Field:
    def __init__(self, n):
        self.n = n

    def __call__(self):
        raise NotImplemented()

    def __len__(self):
        return 1


class PositionField(Field):
    def __init__(self, n, min_coord=0, max_coord=10):
        super(PositionField, self).__init__(n)
        self.min_coord = min_coord
        self.max_coord = max_coord


class IntPositionField(PositionField):
    def __init__(self, n, min_coord=0, max_coord=10, dtype=torch.int):
        super(IntPositionField, self).__init__(n, min_coord, max_coord)
        self.dtype = dtype

    def __call__(self):
        return torch.randint(self.min_coord, self.max_coord, size=(self.n, 1), dtype=self.dtype, requires_grad=False)


class FloatPositionField(PositionField):
    def __init__(self, n, min_coord=0.0, max_coord=1.0, dtype=torch.float):
        super(FloatPositionField, self).__init__(n, min_coord, max_coord)
        self.range = max_coord - min_coord
        self.dtype = dtype

    def __call__(self):
        return (torch.rand(size=(self.n, 1), dtype=self.dtype, requires_grad=False) * self.range) + self.min_coord


# TODO: Do I create a position as a composite of two (or more) position fields, making sure there's no exact overlap?
# TODO: if I do that, I need to make sure the upstream useers can handle a one-dimensional output from here
# TOODO: (rather than zero-d, which is everything so )


class OneHotField(Field):
    def __init__(self, n, n_types, num_per_type=None, dtype=torch.int):
        super(OneHotField, self).__init__(n)
        self.n_types = n_types
        self.num_per_type = num_per_type
        self.dtype = dtype

        if self.num_per_type is not None:
            assert(len(self.num_per_type) == self.n_types)
            assert(sum(self.num_per_type) == n)

    def __call__(self):
        if self.num_per_type is None:
            return self._to_one_hot(torch.randint(self.n_types, size=(self.n,), dtype=self.dtype, requires_grad=False))

        nested_list = [[t] * n for t, n in zip(range(self.n_types), self.num_per_type)]
        flattened_list = [item for sub_list in nested_list for item in sub_list]
        return self._to_one_hot(torch.tensor(flattened_list)[torch.randperm(self.n)])

    def _to_one_hot(self, types):
        one_hot = torch.zeros(self.n, self.n_types, dtype=self.dtype)
        one_hot[torch.arange(self.n, dtype=torch.long), types.to(torch.long)] = 1
        return one_hot

    def __len__(self):
        return self.n_types


FIELD_TYPES = {
    'int_position': IntPositionField,
    'float_position': FloatPositionField,
    'one_hot': OneHotField
}
