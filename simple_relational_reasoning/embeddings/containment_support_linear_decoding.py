import copy
from collections import Counter
import itertools
import typing

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from .models import build_model
from .containment_support_dataset import ContainmentSupportDataset, DecodingDatasets, SCENE_TYPES, DEFAULT_RANDOM_SEED, DEFAULT_VALIDATION_PROPORTION
from .task import METRICS, Metric, TaskResults

BATCH_SIZE = 32
DEFAULT_PATIENCE_EPOCHS = 5
DEFAULT_PATIENCE_MARGIN = 5e-3
DEFAULT_N_TEST_PROPORTION_RANDOM_SEEDS = 5


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
    ) -> typing.Tuple[typing.Dict[str, typing.Any], typing.List[typing.Dict[str, typing.Union[str, int]]]]:
    
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
    per_example_test_correct = []
    for X, y in test_dataloader:
        X = X.to(device)
        y = y.to(device)

        with torch.no_grad():
            embeddings = model(X)
            logits = best_decoder(embeddings)
            loss = criterion(logits, y)
            test_losses.append(loss.item())
            correct = (logits.argmax(dim=1) == y)
            test_accs.append(correct.float().mean().item())
            per_example_test_correct.append(correct.detach().cpu())

    test_loss = np.mean(test_losses)
    test_acc = np.mean(test_accs)
    print(f'With the best model, test loss {test_loss:.4f}, test acc {test_acc:.4f}')

    del train_dataloader
    del val_dataloader
    del test_dataloader
    
    train_min_epoch = np.argmin(train_losses)
    val_min_epoch = np.argmin(val_losses)
    summary_results = dict(
        train_min_loss=train_losses[train_min_epoch], train_min_acc=train_accuracies[train_min_epoch], train_min_epoch=train_min_epoch + 1,
        val_min_loss=val_losses[val_min_epoch], val_min_acc=val_accuracies[val_min_epoch], val_min_epoch=val_min_epoch + 1,
        test_loss=test_loss, test_acc=test_acc, test_epoch=val_min_epoch + 1,
    )

    per_example_test_correct = torch.cat(per_example_test_correct, dim=0).squeeze().numpy()
    for i, config in enumerate(datasets.test_configurations):
        config['correct'] = per_example_test_correct[i]

    return summary_results, datasets.test_configurations


def run_containment_support_linear_decoding_single_model_multiple_features(
    model: nn.Module, dataset: ContainmentSupportDataset, 
    n_epochs: int, lr: float, by_target_object: bool, by_reference_object: bool, 
    test_proportion: typing.Optional[float] = None, n_test_proportion_random_seeds: int = DEFAULT_N_TEST_PROPORTION_RANDOM_SEEDS,
    batch_size: int = BATCH_SIZE, validation_proportion: float = DEFAULT_VALIDATION_PROPORTION,
    patience_epochs: int = DEFAULT_PATIENCE_EPOCHS, patience_margin: float = DEFAULT_PATIENCE_MARGIN,
    random_seed: int = DEFAULT_RANDOM_SEED) -> typing.Tuple[typing.List[typing.Dict[str, typing.Any]], typing.List[typing.Dict[str, typing.Union[int, str]]]]:

    if by_target_object is None and by_reference_object is None and test_proportion is None:
        raise ValueError('test_reference_object, test_target_object, and test_proportion cannot all be None')

    model_results = []
    model_per_example_results = []

    decoding_dataset_kwarg_names = []
    decoding_dataset_kwarg_value_sets = []

    if by_target_object:
        decoding_dataset_kwarg_names.append('test_target_object')
        decoding_dataset_kwarg_value_sets.append(list(dataset.target_objects))

    if by_reference_object:
        decoding_dataset_kwarg_names.append('test_reference_object')
        decoding_dataset_kwarg_value_sets.append(list(dataset.reference_objects))

    if test_proportion is not None:
        decoding_dataset_kwarg_names.append('test_proportion')
        decoding_dataset_kwarg_value_sets.append([test_proportion])
        decoding_dataset_kwarg_names.append('test_seed')
        decoding_dataset_kwarg_value_sets.append(list(range(random_seed, random_seed + n_test_proportion_random_seeds)))

    for value_combination in itertools.product(*decoding_dataset_kwarg_value_sets):
        kwarg_dict = dict(zip(decoding_dataset_kwarg_names, value_combination))
        print(f'Running decoding with {kwarg_dict}')
        
        decoding_datasets = dataset.generate_decoding_datasets(validation_proportion=validation_proportion, **kwarg_dict)
        feature_results, per_example_results = containment_support_linear_decoding_single_model_single_feature(model, decoding_datasets, n_epochs, lr, patience_epochs, patience_margin, batch_size)
        feature_results.update(kwarg_dict)
        for result in per_example_results:
            result.update(kwarg_dict)

        test_type = ''
        if by_target_object:
            test_type += 'target_object'
        if by_reference_object:
            test_type += f'{"_" if len(test_type) else ""}reference_object'
        if test_proportion is not None:
            test_type += f'{"_" if len(test_type) else ""}configuration'
        feature_results['test_type'] = test_type

        for result in per_example_results:
            result['test_type'] = test_type

        model_results.append(feature_results)
        model_per_example_results.extend(per_example_results)

    return model_results, model_per_example_results


def run_containment_support_linear_decoding_multiple_models(
    model_names: typing.Sequence[str], 
    model_kwarg_dicts: typing.Sequence[typing.Dict[str, typing.Any]],
    dataset: ContainmentSupportDataset, 
    n_epochs: int, lr: float, by_target_object: bool, by_reference_object: bool, 
    test_proportion: typing.Optional[float] = None, n_test_proportion_random_seeds: int = DEFAULT_N_TEST_PROPORTION_RANDOM_SEEDS,
    batch_size: int = BATCH_SIZE, validation_proportion: float = DEFAULT_VALIDATION_PROPORTION,
    patience_epochs: int = DEFAULT_PATIENCE_EPOCHS, patience_margin: float = DEFAULT_PATIENCE_MARGIN,
    random_seed: int = DEFAULT_RANDOM_SEED) -> typing.Tuple[typing.List[typing.Dict[str, typing.Any]], typing.List[typing.Dict[str, typing.Union[int, str]]]]:

    all_model_results = []
    all_model_per_example_results = []

    if by_target_object is None and by_reference_object is None and test_proportion is None:
            raise ValueError('test_reference_object, test_target_object, and test_proportion cannot all be None')

    for name, model_kwargs in zip (model_names, model_kwarg_dicts):
        print(f'Starting model {name}')
        model = build_model(**model_kwargs)

        model_results, model_per_example_results = run_containment_support_linear_decoding_single_model_multiple_features(
            model, dataset, n_epochs, lr, by_target_object, by_reference_object, test_proportion, n_test_proportion_random_seeds,
            batch_size, validation_proportion, patience_epochs, patience_margin, random_seed)

        for feature_result in model_results:
            feature_result['model'] = name

        for per_example_result in model_per_example_results:
            per_example_result['model'] = name

        all_model_results.extend(model_results)
        all_model_per_example_results.extend(model_per_example_results)

        del model

        torch.cuda.empty_cache()
        
    return all_model_results, all_model_per_example_results
