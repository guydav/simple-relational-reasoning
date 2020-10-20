from abc import abstractmethod
from collections import defaultdict
from enum import Enum, auto

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader
from simple_relational_reasoning.datagen import QuinnDatasetGenerator


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
    def __init__(self, dataset: QuinnDatasetGenerator,
                 loss=F.cross_entropy, optimizer_class=torch.optim.Adam, lr=1e-4,
                 batch_size=32, train_log_prefix=None, validation_log_prefix=None, test_log_prefix=None):
        super(BaseObjectModel, self).__init__()

        self.dataset = dataset
        sample_input_shape = self.dataset.get_training_dataset().objects[0].shape
        self.object_size = sample_input_shape[1]
        self.num_objects = sample_input_shape[0]

        self.loss = loss
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.batch_size = batch_size

        self.train_log_prefix = train_log_prefix
        self.validation_log_prefix = validation_log_prefix
        self.test_log_prefix = test_log_prefix

    @abstractmethod
    def embed(self, x) -> torch.Tensor:
        pass

    @abstractmethod
    def predict(self, x) -> torch.Tensor:
        pass

    def forward(self, x) -> torch.Tensor:
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

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), self.lr)

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        data, target = batch
        preds = self.forward(data)

        return {'loss': self.loss(preds, target), 'acc': self._compute_accuracy(target, preds)}

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        data, target = batch
        preds = self.forward(data)
        loss_key = f'loss{dataloader_idx if dataloader_idx is not None else ""}'
        acc_key = f'acc{dataloader_idx if dataloader_idx is not None else ""}'

        return {loss_key: self.loss(preds, target), acc_key: self._compute_accuracy(target, preds)}

    # def test_step(self, batch, batch_idx):
    #     return self.training_step(batch, batch_idx)

    def train_dataloader(self):
        return DataLoader(self.dataset.get_training_dataset(), batch_size=self.batch_size)

    def val_dataloader(self):
        # TODO: why is this val_ while the other methods are validation_
        # TODO: this also seems to assume that the dataset is not an iterable one.
        # return DataLoader(self.validation_dataset, batch_size=self.batch_size)
        test_datasets = self.dataset.get_test_datasets()
        return [DataLoader(test_datasets[key], batch_size=self.batch_size)
                for key in sorted(test_datasets.keys())]

    # def test_dataloader(self):
    #     test_datasets = self.dataset.get_test_datasets()
    #     return [DataLoader(test_datasets[key], batch_size=self.batch_size)
    #             for key in sorted(test_datasets.keys())]

    def _average_outputs(self, outputs, prefix, extra_prefix=None):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        logs = {f'{prefix}_loss': avg_loss, f'{prefix}_acc': avg_acc}
        if extra_prefix is not None and len(extra_prefix) > 0:
            logs = {f'{extra_prefix}_{key}': logs[key] for key in logs}

        return logs

    def training_epoch_end(self, outputs):
        return dict(log=(self._average_outputs(outputs, 'train', self.train_log_prefix)))

    def validation_epoch_end(self, outputs):
        val_results = defaultdict(lambda: defaultdict(list))

        for output_list in outputs:
            for output_dict in output_list:
                for key, value  in output_dict.items():
                    key_name, key_idx = key[:-1], int(key[-1])
                    val_results[key_idx][key_name].append(value)

        print('********** TEST EPOCH END: **********')
        print([(key, len(self.dataset.get_test_datasets()[key]))
                for key in sorted(self.dataset.get_test_datasets().keys())])
        print('********** TEST EPOCH END: **********')
        print([(key, len(val_results[key])) for key in val_results])
        print('********** TEST EPOCH END: **********')

        log_dict = {}

        for i, test_set_name in enumerate(sorted(self.dataset.get_test_datasets().keys())):
            log_dict.update(self._average_outputs(val_results[i], test_set_name))

        print(log_dict)
        print('********** TEST EPOCH END: **********')

        return dict(log=log_dict)

    # def test_epoch_end(self, outputs):
    #     print('********** TEST EPOCH END: **********')
    #     print(outputs)
    #     print('********** TEST EPOCH END: **********')
    #     print([len(output_arr) for output_arr in outputs])
    #     print([output_arr.shape for output_arr in outputs])
    #     print('********** TEST EPOCH END: **********')
    #     return dict(log=(self._average_outputs(outputs, 'test', self.test_log_prefix)))

    # def on_epoch_start(self):
    #     if self.regenerate_every_epoch:
    #         self.train_datset.regenerate()
    #         self.validation_dataset.regenerate()
