from abc import abstractmethod
from enum import Enum, auto

import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from simple_relational_reasoning.datagen import ObjectGeneratorDataset


class ObjectCombinationMethod(Enum):
    SUM = auto()
    MEAN = auto()
    CONCAT = auto()

    def combine(self, x, dim=1):
        if self.value == ObjectCombinationMethod.SUM.value:
            return x.sum(dim=dim)

        if self.value == ObjectCombinationMethod.MEAN.value:
            return x.mean(dim=dim)

        if self.value == ObjectCombinationMethod.CONCAT.value:
            return x.view(x.shape[0], -1)


class BaseObjectModel(pl.LightningModule):
    def __init__(self, object_generator, loss=F.cross_entropy, optimizer_class=torch.optim.Adam, lr=1e-4,
                 batch_size=32, train_epoch_size=1024, validation_epoch_size=128, regenerate_every_epoch=False,
                 dataset_class=ObjectGeneratorDataset):
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
        self.train_dataset = dataset_class(self.object_generator, self.train_epoch_size)
        self.validation_dataset = dataset_class(self.object_generator, self.validation_epoch_size)
        self.regenerate_every_epoch = regenerate_every_epoch

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
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)
        # return dict(val_loss=results['loss'], val_acc=results['acc'])

    def val_dataloader(self):
        # TODO: why is this val_ while the other methods are validation_
        # TODO: this also seems to assume that the dataset is not an iterable one.
        return DataLoader(self.validation_dataset, batch_size=self.batch_size)

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        logs = dict(train_loss=avg_loss, train_acc=avg_acc)
        return dict(log=logs)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        logs = dict(val_loss=avg_loss, val_acc=avg_acc)
        return dict(log=logs)

    def on_epoch_start(self):
        if self.regenerate_every_epoch:
            self.train_datset.regenerate()
            self.validation_dataset.regenerate()
