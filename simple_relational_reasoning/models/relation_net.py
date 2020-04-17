import torch
from torch import nn
import torch.nn.functional as F
from simple_relational_reasoning.models.base import BaseObjectModel


DEFAULT_OBJECT_PAIR_LAYER_SIZES = [32, 32, 32]
DEFAULT_COMBINED_OBJECT_LAYER_SIZES = [32, 32]


class RelationNetModel(BaseObjectModel):
    def __init__(self, object_generator,
                 embedding_size=None, embedding_activation_class=nn.ReLU,
                 object_pair_layer_sizes=DEFAULT_OBJECT_PAIR_LAYER_SIZES,
                 object_pair_layer_activation_class=nn.ReLU,
                 combined_object_layer_sizes=DEFAULT_COMBINED_OBJECT_LAYER_SIZES,
                 combined_object_layer_activation_class=nn.ReLU, combined_object_dropout=True,
                 output_size=2, output_activation_class=None,
                 loss=F.cross_entropy, optimizer_class=torch.optim.Adam, lr=1e-4,
                 batch_size=32, train_epoch_size=1024, validation_epoch_size=128):
        super(RelationNetModel, self).__init__(object_generator, loss=loss, optimizer_class=optimizer_class,
                                               lr=lr, batch_size=batch_size, train_epoch_size=train_epoch_size,
                                               validation_epoch_size=validation_epoch_size)

        self.object_input_size = self.object_size
        self.embedding_size = embedding_size
        self.embedding_module = None
        if embedding_size is not None:
            self.embedding_module = nn.Linear(self.object_size, self.embedding_size)
            self.embedding_activation = embedding_activation_class()
            self.object_input_size = embedding_size

        in_size = self.object_input_size * 2
        object_pair_layers = []
        for size in object_pair_layer_sizes:
            object_pair_layers.append(nn.Linear(in_size, size))
            object_pair_layers.append(object_pair_layer_activation_class())
            in_size = size

        self.object_pair_module = nn.Sequential(*object_pair_layers)

        combined_object_layers = []
        for size in combined_object_layer_sizes:
            combined_object_layers.append(nn.Linear(in_size, size))
            combined_object_layers.append(combined_object_layer_activation_class())
            in_size = size

        if combined_object_dropout:
            combined_object_layers.append(nn.Dropout())

        self.combined_object_module = nn.Sequential(*combined_object_layers)

        self.output_size = output_size
        self.output_layer = nn.Linear(in_size, self.output_size)

        if output_activation_class is None:
            output_activation_class = nn.Identity
        self.output_activation = output_activation_class()

    def embed(self, x):
        if self.embedding_module is not None:
            return self.embedding_activation(self.embedding_layer(x))

        return x

    def predict(self, x):
        B, N, E = x.shape

        # TODO: this assumes we also pass in the pairs of each object and itself
        # TODO: is that true? the reference implementations appear to do so
        x_i = x.unsqueeze(1).repeat(1, N, 1, 1)  # B, N, N, E
        x_j = x.unsqueeze(2).repeat(1, 1, N, 1)  # B, N, N, E
        # this ends up tiling them opposite of each other
        x = torch.cat([x_i, x_j], 3)  # B, N, N, 2E

        x = self.object_pair_module(x)  # B, N, N, P = output size of object pair module
        x = x.sum(2).sum(1)  # B, P => one embedding per n^2 object pairs

        x = self.combined_object_module(x)  # B, C = output size of combined object module
        return self.output_activation(self.output_layer(x))

