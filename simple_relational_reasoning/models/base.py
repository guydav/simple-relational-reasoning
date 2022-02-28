from abc import abstractmethod
from collections import defaultdict
from enum import Enum, auto

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader
from simple_relational_reasoning.datagen import QuinnBaseDatasetGenerator


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
    def __init__(self, dataset: QuinnBaseDatasetGenerator,
                 loss=F.cross_entropy, optimizer_class=torch.optim.Adam, lr=1e-4,
                 batch_size=32, train_log_prefix=None, validation_log_prefix=None, test_log_prefix=None):
        super(BaseObjectModel, self).__init__()

        self.dataset = dataset
        train = self.dataset.get_training_dataset()
        self.object_size = train.get_object_size()
        self.num_objects = train.get_num_objects()
        self.output_size = train.get_num_classes()

        self.loss = loss
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.batch_size = batch_size

        self.train_log_prefix = train_log_prefix
        self.validation_log_prefix = validation_log_prefix
        self.test_log_prefix = test_log_prefix

        self.val_dataloader_names = []

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
        loss = self.loss(preds, target)
        self.log('train_loss', self.loss(preds, target), on_step=False, on_epoch=True)
        self.log('train_acc', self._compute_accuracy(target, preds), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        data, target = batch
        preds = self.forward(data)
        self.log(f'{self.val_dataloader_names[dataloader_idx]}_loss', self.loss(preds, target), 
            on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f'{self.val_dataloader_names[dataloader_idx]}_acc', self._compute_accuracy(target, preds), 
            on_step=False, on_epoch=True, add_dataloader_idx=False)
        
    # def test_step(self, batch, batch_idx, dataloader_idx=None):
    #     data, target = batch
    #     preds = self.forward(data)
    #     loss_key = f'loss{dataloader_idx if dataloader_idx is not None else ""}'
    #     acc_key = f'acc{dataloader_idx if dataloader_idx is not None else ""}'
    #     self.log('test_loss', self.loss(preds, target), on_step=False, on_epoch=True)
    #     self.log('test_acc', self._compute_accuracy(target, preds), on_step=False, on_epoch=True)

    def train_dataloader(self):
        train = self.dataset.get_training_dataset()
        print(f'TRAINING SET SIZE: {len(train)}')
        return DataLoader(train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        dataloaders = []

        val_dataset = self.dataset.get_validation_dataset()
        if val_dataset is not None:
            print(f'VALIDATION SET SIZE: {len(val_dataset)}')
            dataloaders.append(DataLoader(val_dataset, shuffle=False, batch_size=self.batch_size))
            self.val_dataloader_names.append('val')

        test_datasets = self.dataset.get_test_datasets()
        for key in sorted(test_datasets.keys()):
            dataloaders.append(DataLoader(test_datasets[key], shuffle=False, batch_size=self.batch_size))
            self.val_dataloader_names.append(key)

        return dataloaders

    # def test_dataloader(self):
        

    # def _average_outputs(self, outputs, prefix, extra_prefix=None):
    #     avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    #     avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
    #     logs = {f'{prefix}_loss': avg_loss, f'{prefix}_acc': avg_acc}
    #     if extra_prefix is not None and len(extra_prefix) > 0:
    #         logs = {f'{extra_prefix}_{key}': logs[key] for key in logs}

    #     return logs

    # def training_epoch_end(self, outputs):
    #     return self.log(self._average_outputs(outputs, 'train', self.train_log_prefix))

    # def validation_epoch_end(self, outputs):
    #     return self.log(self._average_outputs(outputs, 'val', self.validation_log_prefix))

    # def test_epoch_end(self, outputs):
    #     test_results = defaultdict(lambda: defaultdict(list))

    #     for output_list in outputs:
    #         for output_dict in output_list:
    #             for key, value in output_dict.items():
    #                 key_name, key_idx = key[:-1], int(key[-1])
    #                 test_results[key_idx][key_name].append(value)

    #     # print('********** TEST EPOCH END: **********')
    #     # print([(key, len(self.dataset.get_test_datasets()[key]))
    #     #         for key in sorted(self.dataset.get_test_datasets().keys())])
    #     # print('********** TEST EPOCH END: **********')
    #     # print(val_results)
    #     # print('********** TEST EPOCH END: **********')

    #     log_dict = {}

    #     for i, test_set_name in enumerate(sorted(self.dataset.get_test_datasets().keys())):
    #         log_dict.update({f'{test_set_name}_{key}': torch.stack(test_results[i][key]).mean()
    #                         for key in test_results[i]})

    #     # print(log_dict)
    #     # print('********** TEST EPOCH END: **********')

    #     self.log(log_dict)

    # def on_validation_epoch_end(self):
    #     print('On epoch end called')
    #     self.trainer.test(self)

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
