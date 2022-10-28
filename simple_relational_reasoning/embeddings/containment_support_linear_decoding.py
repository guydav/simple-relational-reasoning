import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import typing

import copy
from collections import defaultdict, namedtuple
from IPython.display import display, Markdown
from tqdm.notebook import tqdm

from .models import build_model
from .containment_support_dataset import ContainmentSupportDataset, DecodingDatasets, SCENE_TYPES, DEFAULT_RANDOM_SEED, DEFAULT_VALIDATION_PROPORTION

import os

from .task import METRICS, Metric, TaskResults

BATCH_SIZE = 32


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


def containment_support_linear_decoding_single_model_single_feature(
    model: nn.Module, datasets: DecodingDatasets, 
    n_epochs: int, lr: float,
    batch_size: int = BATCH_SIZE, 
    device: typing.Optional[torch.device] = None, 
    ):
    
    if device is None:
        device = next(model.parameters()).device

    B = batch_size

    train_dataloader = DataLoader(datasets.train, batch_size=B, shuffle=True)
    val_dataloader = DataLoader(datasets.val, batch_size=B, shuffle=False)
    test_dataloader = DataLoader(datasets.test, batch_size=B, shuffle=False)

    decoder = LinearClassifier(model.embedding_dim, datasets.n_classes).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    best_decoder = None
    min_val_loss = np.inf

    for epoch in range(n_epochs):
        epoch_train_losses = []
        epoch_train_accs = []

        decoder.train()

        for X, y in train_dataloader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                embeddings = model(X)

            logits = decoder(embeddings)
            loss = criterion(logits, y)
            epoch_train_losses.append(loss.item())
            epoch_train_accs.append((logits.argmax(dim=1) == y).float().mean())

            loss.backward()
            optimizer.step()
        
        epoch_train_loss = np.mean(epoch_train_losses)
        epoch_train_acc = np.mean(epoch_train_accs)
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        decoder.eval()

        epoch_val_losses = []
        epoch_val_accs = []

        for X, y in val_dataloader:
            X = X.to(device)
            y = y.to(device)

            with torch.no_grad():
                embeddings = model(X)
                logits = decoder(embeddings)
                loss = criterion(logits, y)
                epoch_val_losses.append(loss.item())
                epoch_val_accs.append((logits.argmax(dim=1) == y).float().mean())

        epoch_val_loss = np.mean(epoch_val_losses)
        epoch_val_acc = np.mean(epoch_val_accs)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        print(f'After epoch {epoch}: train loss {epoch_train_loss:.3f}, train acc {epoch_train_acc:.3f}, val loss {epoch_val_loss:.3f}, val acc {epoch_val_acc:.3f}')

        if epoch_val_loss < min_val_loss:
            print(f'New best val loss {epoch_val_loss:.3f} < {min_val_loss:.3f}, copying decoder')
            min_val_loss = epoch_val_loss
            best_decoder = copy.deepcopy(decoder).cpu()

    best_decoder = typing.cast(nn.Module, best_decoder).to(device).eval()

    test_losses = []
    test_accs = []
    for X, y in test_dataloader:
        X = X.to(device)
        y = y.to(device)

        with torch.no_grad():
            embeddings = model(X)
            logits = best_decoder(embeddings)
            loss = criterion(logits, y)
            test_losses.append(loss.item())
            test_accs.append((logits.argmax(dim=1) == y).float().mean())

    test_loss = np.mean(test_losses)
    test_acc = np.mean(test_accs)
    print(f'With the best model, test loss {test_loss:.3f}, test acc {test_acc:.3f}')

    del train_dataloader
    del val_dataloader
    del test_dataloader
    
    train_min_epoch = np.argmin(train_losses)
    val_min_epoch = np.argmin(val_losses)
    return dict(
        train_min_loss=train_losses[train_min_epoch], train_min_acc=train_accuracies[train_min_epoch], train_min_epoch=train_min_epoch + 1,
        val_min_loss=val_losses[val_min_epoch], val_min_acc=val_accuracies[val_min_epoch], val_min_epoch=val_min_epoch + 1,
        test_loss=test_loss, test_acc=test_acc, test_epoch=val_min_epoch + 1,
    )


def run_containment_support_linear_decoding_single_model_multiple_features(
    model: nn.Module, dataset: ContainmentSupportDataset, 
    n_epochs: int, lr: float, by_target_object: bool, by_reference_object: bool,
    batch_size: int = BATCH_SIZE, validation_proportion: float = DEFAULT_VALIDATION_PROPORTION,
    random_seed: int = DEFAULT_RANDOM_SEED, ):

    if by_target_object == by_reference_object:
        raise ValueError('Exactly one of by_target_object and by_reference_object must be True')

    model_results = []

    if by_target_object:
        for target_object in dataset.target_objects:
            decoding_datasets = dataset.generate_decoding_datasets(test_target_object=target_object, validation_proportion=validation_proportion, random_seed=random_seed)
            feature_results = containment_support_linear_decoding_single_model_single_feature(model, decoding_datasets, n_epochs, lr, batch_size)
            feature_results['test_target_object'] = target_object
            feature_results['test_type'] = 'target_object'
            model_results.append(feature_results)

    else:
        for reference_object in dataset.reference_objects:
            decoding_datasets = dataset.generate_decoding_datasets(test_reference_object=reference_object, validation_proportion=validation_proportion, random_seed=random_seed)
            feature_results = containment_support_linear_decoding_single_model_single_feature(model, decoding_datasets, n_epochs, lr, batch_size)
            feature_results['test_reference_object'] = reference_object
            feature_results['test_type'] = 'reference_object'
            model_results.append(feature_results)

    return model_results


def run_containment_support_linear_decoding_multiple_models(
    model_names: typing.Sequence[str], 
    model_kwarg_dicts: typing.Sequence[typing.Dict[str, typing.Any]],
    dataset: ContainmentSupportDataset, 
    n_epochs: int, lr: float, by_target_object: bool, by_reference_object: bool,
    batch_size: int = BATCH_SIZE, validation_proportion: float = DEFAULT_VALIDATION_PROPORTION,
    random_seed: int = DEFAULT_RANDOM_SEED, ):

    all_model_results = []
    
    for name, model_kwargs in zip (model_names, model_kwarg_dicts):
        print(f'Starting model {name}')
        model = build_model(**model_kwargs)

        model_results = run_containment_support_linear_decoding_single_model_multiple_features(
            model, dataset, n_epochs, lr, by_target_object, by_reference_object, batch_size, validation_proportion, random_seed)

        for feature_result in model_results:
            feature_result['model'] = name

        all_model_results.extend(model_results)

        del model

        torch.cuda.empty_cache()
        
    return all_model_results






    if by_reference_object == by_target_object:
        raise ValueError('Must set exactly one of by_reference_object and by_target_object to True')