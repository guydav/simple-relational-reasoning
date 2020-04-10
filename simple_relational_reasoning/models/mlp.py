import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from simple_relational_reasoning.datagen import ObjectGeneratorDataset


class MLPAverageModel(pl.LightningModule):
    def __init__(self, object_generator, representation_size, representation_activation,
                 prediction_size=1, prediction_activation=torch.sigmoid,
                 loss=F.mse_loss, optimizer_class=torch.optim.Adam, lr=1e-4,
                 batch_size=32, train_epoch_size=1024, validation_epoch_size=128):
        super(MLPAverageModel, self).__init__()

        self.object_generator = object_generator

        self.representation_layer = nn.Linear(self.object_generator.object_size, representation_size)
        self.representation_activation = representation_activation

        self.prediction_layer = nn.Linear(representation_size, prediction_size)
        self.prediction_activation = prediction_activation

        self.loss = loss
        self.optimizer_class = optimizer_class
        self.lr = lr

        self.batch_size = batch_size
        self.train_epoch_size = train_epoch_size
        self.validation_epoch_size = validation_epoch_size
        self.train_datset = ObjectGeneratorDataset(self.object_generator, self.train_epoch_size)
        self.validation_dataset = ObjectGeneratorDataset(self.object_generator, self.validation_epoch_size)

    def forward(self, x):
        """
        :param x: a batch, expected to be of shape (B, N, F): B sets per batch, N objects per set, F features per object
        :return: The prediction for each object
        """
        B, N, F = x.shape
        x = x.reshape(B * N, F)
        representations = self.representation_activation(self.representation_layer(x))
        # TODO: verify this reshapes correctly
        reshaped_representations = representations.reshape(B, N, -1)
        # will be (B, self.representation_size)
        set_representations = reshaped_representations.mean(1)
        # will be (B, self.prediction_size)
        return self.prediction_activation(self.prediction_layer(set_representations))

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self.forward(data).squeeze()
        return dict(loss=self.loss(preds, target), acc=(target == torch.round(preds)).float().mean())

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
