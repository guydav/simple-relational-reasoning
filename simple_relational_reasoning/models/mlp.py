import torch
from torch import nn
import torch.nn.functional as F
from simple_relational_reasoning.models.base import BaseObjectModel, ObjectCombinationMethod


class MLPModel(BaseObjectModel):
    def __init__(self, dataset, embedding_size, embedding_activation_class,
                 prediction_sizes=None, prediction_activation_class=None,
                 output_activation_class=None, loss=F.cross_entropy, optimizer_class=torch.optim.Adam, lr=1e-4,
                 batch_size=32, train_log_prefix=None, validation_log_prefix=None, test_log_prefix=None):
        super(MLPModel, self).__init__(dataset, oss=loss, optimizer_class=optimizer_class,
                                       lr=lr, batch_size=batch_size,
                                       train_log_prefix=train_log_prefix,
                                       validation_log_prefix=validation_log_prefix,
                                       test_log_prefix=test_log_prefix)

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


class CombinedObjectMLPModel(BaseObjectModel):
    def __init__(self, dataset,
                 embedding_size=None, embedding_activation_class=nn.ReLU,
                 object_combiner=ObjectCombinationMethod.MEAN,
                 prediction_sizes=None, prediction_activation_class=nn.ReLU,
                 output_activation_class=None, loss=F.cross_entropy, optimizer_class=torch.optim.Adam, lr=1e-4,
                 batch_size=32, train_log_prefix=None, validation_log_prefix=None, test_log_prefix=None):
        super(CombinedObjectMLPModel, self).__init__(dataset, loss=loss, optimizer_class=optimizer_class,
                                                     lr=lr, batch_size=batch_size,
                                                     train_log_prefix=train_log_prefix,
                                                     validation_log_prefix=validation_log_prefix,
                                                     test_log_prefix=test_log_prefix)

        self.embedding_size = embedding_size
        self.embedding_module = nn.Identity()

        if self.embedding_size is not None:
            self.embedding_module = nn.Sequential(
                nn.Linear(self.object_size, self.embedding_size),
                embedding_activation_class()
            )

        self.object_combiner = object_combiner

        output_layer_input_size = (self.embedding_size is not None) and self.embedding_size or self.object_size
        if self.object_combiner == ObjectCombinationMethod.CONCAT:
            output_layer_input_size *= self.num_objects

        self.prediction_module = nn.Identity()

        if prediction_sizes is not None:
            prediction_layers = []
            for size in prediction_sizes:
                prediction_layers.append(nn.Linear(output_layer_input_size, size))
                prediction_layers.append(prediction_activation_class())
                output_layer_input_size = size

            self.prediction_module = nn.Sequential(*prediction_layers)

        self.output_layer = nn.Linear(output_layer_input_size, self.output_size)

        if output_activation_class is None:
            output_activation_class = nn.Identity
        self.output_activation = output_activation_class()

    def embed(self, x):
        return self.embedding_module(x)

    def predict(self, x):
        x = self.object_combiner.combine(x)
        x = self.prediction_module(x)
        return self.output_activation(self.output_layer(x))
