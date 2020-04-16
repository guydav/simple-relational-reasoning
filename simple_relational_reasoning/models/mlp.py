import torch
from torch import nn
import torch.nn.functional as F
from simple_relational_reasoning.models.base import BaseObjectModel


class MLPModel(BaseObjectModel):
    def __init__(self, object_generator, embedding_size, embedding_activation_class,
                 prediction_sizes=None, prediction_activation_class=None,
                 output_size=2, output_activation_class=None,
                 loss=F.cross_entropy, optimizer_class=torch.optim.Adam, lr=1e-4,
                 batch_size=32, train_epoch_size=1024, validation_epoch_size=128):
        super(MLPModel, self).__init__(object_generator, loss=loss, optimizer_class=optimizer_class,
                                       lr=lr, batch_size=batch_size, train_epoch_size=train_epoch_size,
                                       validation_epoch_size=validation_epoch_size)

        self.embedding_size = embedding_size
        self.embedding_module = nn.Linear(self.object_size, self.embedding_size)
        self.embedding_activation = embedding_activation_class()

        output_layer_input_size = self.embedding_size * self.num_objects

        self.prediction_module = None
        if prediction_sizes is not None:
            in_size = output_layer_input_size
            prediction_layers = []
            for size in prediction_sizes:
                prediction_layers.append(nn.Linear(in_size, size))
                prediction_layers.append(prediction_activation_class())
                in_size = size

            self.prediction_module = nn.Sequential(*prediction_layers)
            output_layer_input_size = in_size

        self.output_size = output_size
        self.output_layer = nn.Linear(output_layer_input_size, self.output_size)

        if output_activation_class is None:
            output_activation_class = nn.Identity
        self.output_activation = output_activation_class()

    def embed(self, x):
        return self.embedding_activation(self.embedding_module(x))

    def predict(self, x):
        x = x.view(x.shape[0], -1)
        if self.prediction_module is not None:
            x = self.prediction_module(x)
        return self.output_activation(self.output_layer(x))

