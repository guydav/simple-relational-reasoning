import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import typing

from abc import abstractmethod
from collections import defaultdict, namedtuple
from IPython.display import display, Markdown
from tqdm.notebook import tqdm

from .models import build_model
from .containment_support_dataset import ContainmentSupportDataset, SCENE_TYPES

import os

from .task import METRICS, Metric, TaskResults

BATCH_SIZE = 32


def containment_support_task_single_model(
    model: nn.Module, dataset: ContainmentSupportDataset, metrics: typing.Sequence[Metric]=METRICS, 
    batch_size: int = BATCH_SIZE, use_tqdm: bool = False, 
    device: typing.Optional[torch.device] = None, tsne_mode: bool = False, aggregate_results: bool = False
    ):
    
    if device is None:
        device = next(model.parameters()).device

    B = batch_size
    dataloader = DataLoader(TensorDataset(dataset.dataset), batch_size=batch_size, shuffle=False)
    
    model_results = defaultdict(list)
    cos = nn.CosineSimilarity(dim=-1)
    n_stimuli_per_batch = len(dataset.scene_types)
    triangle_indices = np.triu_indices(n_stimuli_per_batch, 1)
    
    model.eval()
    
    data_iter = dataloader
    if use_tqdm:
        data_iter = tqdm(dataloader, desc='Batches')

    all_results = []
    output_embeddings = []

    for b in data_iter:
        if tsne_mode:
            x = b[0]  # shape (B, K, 3, 224, 224) 
            B, K = x.shape[:2]
            x = x.view(-1, *x.shape[2:])
            x = x.to(device)
            e = model(x).detach().view(B, K, -1).cpu().numpy()
            output_embeddings.append(e)

        else:
            x = b[0]  # shape (B, 4, 3, 224, 224) where H is the number of habituation stimuli
            x = x.view(-1, *x.shape[2:])
            x = x.to(device)
            e = model(x).detach()
            e = e.view(B, n_stimuli_per_batch, -1)  # shape (B, H + 2, Z)

            embedding_pairwise_cosine = cos(e[:, :, None, :], e[:, None, :, :])  # shape (B, K , K) where K = 3 for 3 stimuli and K = 6 for 4 stimuli
            triplet_cosines = embedding_pairwise_cosine[:, triangle_indices[0], triangle_indices[1]] # shape (B, K)

            if aggregate_results:
                for metric in metrics:
                    model_results[metric.name].append(metric(triplet_cosines).cpu().numpy())
            else:
                all_results.append(triplet_cosines.cpu())

    if aggregate_results and not tsne_mode:
        for metric in metrics:
            model_results[metric.name] = metric.aggregate(model_results[metric.name])  # type: ignore

    del dataloader

    # if tsne_mode:
    #     return data.cpu().numpy(), np.concatenate(output_embeddings, axis=0)
    # del data
    
    if tsne_mode:
        return np.concatenate(output_embeddings, axis=0)

    if not aggregate_results:
        return np.concatenate(all_results, axis=0)

    return model_results


def run_containment_support_task_multiple_models(
        model_names: typing.Sequence[str], 
        model_kwarg_dicts: typing.Sequence[typing.Dict[str, typing.Any]],
        dataset: ContainmentSupportDataset,
        batch_size: int = BATCH_SIZE, tsne_mode: bool = False, aggregate_results: bool = False):

    all_model_results = {}
    
    for name, model_kwargs in zip (model_names, model_kwarg_dicts):
        print(f'Starting model {name}')
        model = build_model(**model_kwargs)

        all_model_results[name] = containment_support_task_single_model(
            model, dataset, batch_size=batch_size, tsne_mode=tsne_mode, aggregate_results=aggregate_results)

        del model

        torch.cuda.empty_cache()
        
    return all_model_results

