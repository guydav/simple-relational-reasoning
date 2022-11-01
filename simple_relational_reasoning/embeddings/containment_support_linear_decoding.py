import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import typing

import copy
from collections import Counter
from IPython.display import display, Markdown
from tqdm.notebook import tqdm

from .models import build_model
from .containment_support_dataset import ContainmentSupportDataset, DecodingDatasets, SCENE_TYPES, DEFAULT_RANDOM_SEED, DEFAULT_VALIDATION_PROPORTION

import os

from .task import METRICS, Metric, TaskResults

BATCH_SIZE = 32
DEFAULT_PATIENCE_EPOCHS = 5
DEFAULT_PATIENCE_MARGIN = 1e-3


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


def _weigted_random_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    numpy_labels = labels.numpy()
    class_counts = Counter(numpy_labels)
    total = len(numpy_labels)
    return WeightedRandomSampler([total / class_counts[y] for y in numpy_labels], total)


def containment_support_linear_decoding_single_model_single_feature(
    model: nn.Module, datasets: DecodingDatasets, 
    n_epochs: int, lr: float,
    patience_epochs: int = DEFAULT_PATIENCE_EPOCHS, patience_margin: float = DEFAULT_PATIENCE_MARGIN,
    batch_size: int = BATCH_SIZE, 
    device: typing.Optional[torch.device] = None, 
    ):
    
    if device is None:
        device = next(model.parameters()).device

    B = batch_size

    train_dataloader = DataLoader(datasets.train, batch_size=B, num_workers=4, sampler=_weigted_random_sampler(datasets.train.tensors[1]))
    val_dataloader = DataLoader(datasets.val, batch_size=B, num_workers=1, shuffle=False)
    test_dataloader = DataLoader(datasets.test, batch_size=B, num_workers=1, shuffle=False)

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
    patience_val_loss = np.inf
    patience_val_epoch = np.inf

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
            epoch_train_accs.append((logits.argmax(dim=1) == y).float().mean().item())

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
                epoch_val_accs.append((logits.argmax(dim=1) == y).float().mean().item())

        epoch_val_loss = np.mean(epoch_val_losses)
        epoch_val_acc = np.mean(epoch_val_accs)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        if epoch_val_loss < min_val_loss:
            print(f'After epoch {epoch}: train acc {epoch_train_acc:.4f}, train loss {epoch_train_loss:.4f}, val acc {epoch_val_acc:.4f}, val loss {epoch_val_loss:.4f} < {min_val_loss:.4f}, copying decoder')
            min_val_loss = epoch_val_loss
            best_decoder = copy.deepcopy(decoder).cpu()

        else:
            print(f'After epoch {epoch}: train acc {epoch_train_acc:.4f}, train loss {epoch_train_loss:.4f}, val acc {epoch_val_acc:.4f}, val loss {epoch_val_loss:.4f}')

        if epoch_val_loss < patience_val_loss - patience_margin:
            patience_val_loss = epoch_val_loss
            patience_val_epoch = epoch

        if epoch - patience_val_epoch >= patience_epochs:
            print(f'Insufficient improvement in val loss for {patience_epochs} epochs, stopping')
            break

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
            test_accs.append((logits.argmax(dim=1) == y).float().mean().item())

    test_loss = np.mean(test_losses)
    test_acc = np.mean(test_accs)
    print(f'With the best model, test loss {test_loss:.4f}, test acc {test_acc:.4f}')

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
    test_proportion: typing.Optional[float] = None,
    batch_size: int = BATCH_SIZE, validation_proportion: float = DEFAULT_VALIDATION_PROPORTION,
    patience_epochs: int = DEFAULT_PATIENCE_EPOCHS, patience_margin: float = DEFAULT_PATIENCE_MARGIN,
    random_seed: int = DEFAULT_RANDOM_SEED, ):

    if by_target_object is None and by_reference_object is None and test_proportion is None:
            raise ValueError('test_reference_object, test_target_object, and test_proportion cannot all be None')

    if int(by_target_object) + int(by_reference_object) + int(test_proportion is not None) > 1:
        raise ValueError('Only one of test_reference_object, test_target_object, and test_proportion can be specified')

    model_results = []

    if by_target_object:
        for target_object in dataset.target_objects:
            print(f'Starting target object {target_object}')
            decoding_datasets = dataset.generate_decoding_datasets(test_target_object=target_object, validation_proportion=validation_proportion, 
                test_proportion=test_proportion, random_seed=random_seed)
            feature_results = containment_support_linear_decoding_single_model_single_feature(model, decoding_datasets, n_epochs, lr, patience_epochs, patience_margin, batch_size)
            feature_results['test_target_object'] = target_object
            feature_results['test_type'] = 'target_object'
            model_results.append(feature_results)

    else:
        for reference_object in dataset.reference_objects:
            print(f'Starting reference object {reference_object}')
            decoding_datasets = dataset.generate_decoding_datasets(test_reference_object=reference_object, validation_proportion=validation_proportion, 
                test_proportion=test_proportion, random_seed=random_seed)
            feature_results = containment_support_linear_decoding_single_model_single_feature(model, decoding_datasets, n_epochs, lr, patience_epochs, patience_margin, batch_size)
            feature_results['test_reference_object'] = reference_object
            feature_results['test_type'] = 'reference_object'
            model_results.append(feature_results)

    return model_results


def run_containment_support_linear_decoding_multiple_models(
    model_names: typing.Sequence[str], 
    model_kwarg_dicts: typing.Sequence[typing.Dict[str, typing.Any]],
    dataset: ContainmentSupportDataset, 
    n_epochs: int, lr: float, by_target_object: bool, by_reference_object: bool, test_proportion: typing.Optional[float] = None,
    batch_size: int = BATCH_SIZE, validation_proportion: float = DEFAULT_VALIDATION_PROPORTION,
    patience_epochs: int = DEFAULT_PATIENCE_EPOCHS, patience_margin: float = DEFAULT_PATIENCE_MARGIN,
    random_seed: int = DEFAULT_RANDOM_SEED, ):

    all_model_results = []

    if by_target_object is None and by_reference_object is None and test_proportion is None:
            raise ValueError('test_reference_object, test_target_object, and test_proportion cannot all be None')

    if int(by_target_object) + int(by_reference_object) + int(test_proportion is not None) != 1:
        raise ValueError('Only one of test_reference_object, test_target_object, and test_proportion can be specified')

    
    for name, model_kwargs in zip (model_names, model_kwarg_dicts):
        print(f'Starting model {name}')
        model = build_model(**model_kwargs)

        model_results = run_containment_support_linear_decoding_single_model_multiple_features(
            model, dataset, n_epochs, lr, by_target_object, by_reference_object, test_proportion, 
            batch_size, validation_proportion, patience_epochs, patience_margin, random_seed)

        for feature_result in model_results:
            feature_result['model'] = name

        all_model_results.extend(model_results)

        del model

        torch.cuda.empty_cache()
        
    return all_model_results
