from abc import abstractmethod

import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from simple_relational_reasoning.datagen import ObjectGeneratorDataset


class BaseObjectModel(pl.LightningModule):
    def __init__(self, object_generator, loss=F.cross_entropy, optimizer_class=torch.optim.Adam, lr=1e-4,
                 batch_size=32, train_epoch_size=1024, validation_epoch_size=128):
        super(BaseObjectModel, self).__init__()

        self.object_generator = object_generator
        self.object_size = self.object_generator.object_size
        self.num_objects = self.object_generator.n

        self.loss = loss
        self.optimizer_class = optimizer_class
        self.lr = lr

        self.batch_size = batch_size
        self.train_epoch_size = train_epoch_size
        self.validation_epoch_size = validation_epoch_size
        self.train_datset = ObjectGeneratorDataset(self.object_generator, self.train_epoch_size)
        self.validation_dataset = ObjectGeneratorDataset(self.object_generator, self.validation_epoch_size)

    @abstractmethod
    def embed(self, x):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    def forward(self, x):
        """
        :param x: a batch, expected to be of shape (B, N, F): B sets per batch, N objects per set, F features per object
        :return: The prediction for each object
        """
        # B, N, F = x.shape
        # After the next call, x should be of shape (B, N, E) where E is the embedding size
        x = self.embed(x)
        # The returned value should be of shape (B, L), where L is the shape the loss function expects
        return self.predict(x)

    def _compute_accuracy(self, target, preds):
        return torch.eq(target, preds.argmax(1)).float().mean()

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self.forward(data)
        return dict(loss=self.loss(preds, target), acc=self._compute_accuracy(target, preds))

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), self.lr)

    def train_dataloader(self):
        return DataLoader(ObjectGeneratorDataset(self.object_generator, self.train_epoch_size),
                          batch_size=self.batch_size)

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        return dict(val_loss=results['loss'], val_acc=results['acc'])

    def val_dataloader(self):
        # TODO: why is this val_ while the other methods are validation_
        # TODO: this also seems to assume that the dataset is not an iterable one.
        return DataLoader(self.validation_dataset, batch_size=self.batch_size)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        logs = dict(val_loss=avg_loss, val_acc=avg_acc)
        print(logs)
        return logs

    def on_epoch_start(self):
        self.train_datset.regenerate()
        self.validation_dataset.regenerate()


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
        self.embedding_layer = nn.Linear(self.object_size, self.embedding_size)
        self.embedding_activation = embedding_activation_class

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
        return self.embedding_activation(self.embedding_layer(x))

    def predict(self, x):
        x = x.view(x.shape[0], -1)
        if self.prediction_module is not None:
            x = self.prediction_module(x)
        return self.output_activation(self.output_layer(x))

