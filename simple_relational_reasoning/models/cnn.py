import torch
from torch import nn
import torch.nn.functional as F
from simple_relational_reasoning.models.base import BaseObjectModel
from simple_relational_reasoning.datagen import SpatialObjectGeneratorDataset


DEFAULT_CONV_SIZES = [16, 32]
DEFAULT_MLP_SIZES = [32, 32]


class CNNModel(BaseObjectModel):
    def __init__(self, dataset,
                 conv_sizes=DEFAULT_CONV_SIZES, conv_activation_class=nn.ReLU,
                 conv_kernel_size=3, conv_stride=1, conv_padding=1, conv_input_size=None,
                 mlp_sizes=DEFAULT_MLP_SIZES, mlp_activation_class=nn.ReLU, output_activation_class=None,
                 loss=F.cross_entropy, optimizer_class=torch.optim.Adam, lr=1e-4,
                 batch_size=32, train_log_prefix=None, validation_log_prefix=None, test_log_prefix=None):
        super(CNNModel, self).__init__(dataset, loss=loss,
                                       optimizer_class=optimizer_class,
                                       lr=lr, batch_size=batch_size,
                                       train_log_prefix=train_log_prefix,
                                       validation_log_prefix=validation_log_prefix,
                                       test_log_prefix=test_log_prefix)

        if hasattr(conv_kernel_size, '__len__') and len(conv_kernel_size) != len(conv_sizes):
            raise ValueError(f'The length of kernel sizes provided {conv_kernel_size} must be the same as the length of the conv sizes {conv_sizes}')

        else:
            conv_kernel_size = [conv_kernel_size] * len(conv_sizes)

        if hasattr(conv_stride, '__len__') and len(conv_stride) != len(conv_sizes):
            raise ValueError(f'The length of strides provided {conv_stride} must be the same as the length of the conv sizes {conv_sizes}')

        else:
            conv_stride = [conv_stride] * len(conv_sizes)

        if hasattr(conv_padding, '__len__') and len(conv_padding) != len(conv_sizes):
            raise ValueError(f'The length of paddings provided {conv_padding} must be the same as the length of the conv sizes {conv_sizes}')

        else:
            conv_padding = [conv_padding] * len(conv_sizes)

        conv_input_size = self.object_size
        conv_layers = []
        for size, kernel_size, stride, padding in zip(conv_sizes, conv_kernel_size, conv_stride, conv_padding):
            conv_layers.append(nn.Conv2d(in_channels=conv_input_size, out_channels=size, kernel_size=kernel_size,
                                         stride=stride, padding=padding))

            conv_layers.append(conv_activation_class())
            conv_layers.append(nn.MaxPool2d(2))
            # TODO: normalization? pooling?
            conv_input_size = size

        # remove the last pooling layer to replace it with an average pool, to get some degree of invariance
        conv_layers[-1] = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_module = nn.Sequential(*conv_layers)

        mlp_input_size = conv_sizes[-1]
        mlp_layers = []
        for size in mlp_sizes:
            mlp_layers.append(nn.Linear(mlp_input_size, size))
            mlp_layers.append(mlp_activation_class())
            mlp_input_size = size

        self.mlp_module = nn.Sequential(*mlp_layers)

        self.output_layer = nn.Linear(mlp_input_size, self.output_size)

        if output_activation_class is None:
            output_activation_class = nn.Identity
        self.output_activation = output_activation_class()

    def embed(self, x):
        return x

    def predict(self, x):
        x = self.conv_module(x)

        x = x.view(x.shape[0], -1)
        x = self.mlp_module(x)
        return self.output_activation(self.output_layer(x))


class SimplifiedCNNModel(CNNModel):
    pass
