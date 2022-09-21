import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import tabulate
import typing

from abc import abstractmethod
from collections import defaultdict, namedtuple
from IPython.display import display, Markdown
from tqdm.notebook import tqdm

from .models import build_model

import os


class Metric:
    name: str
    correct_index : int
    def __init__(self, name: str, correct_index: int = 0):
        self.name = name
        self.correct_index = correct_index
        
    @abstractmethod
    def __call__(self, pairwise_cosines: typing.Union[torch.Tensor, np.ndarray]) -> typing.Union[torch.Tensor, np.ndarray]:
        raise NotImplementedError()
    
    def aggregate(self, result_list) -> typing.Union[torch.Tensor, np.ndarray]:
        if isinstance(result_list[0], torch.Tensor):
            return torch.cat(result_list).detach().cpu().numpy()
        
        if isinstance(result_list[0], np.ndarray):
            return np.concatenate(result_list)  # type: ignore
        
        raise ValueError(f'Can only combine lists of torch.Tensor or np.ndarray, received {type(result_list[0])}')

        
class AccuracyMetric(Metric):
    pair_only: bool
    pair_comparison_index: int
    def __init__(self, name: str, correct_index: int = 0, pair_only: bool = False, pair_comparison_index: int = 1):
        super(AccuracyMetric, self).__init__(name, correct_index)
        self.pair_only = pair_only
        self.pair_comparison_index = pair_comparison_index
        
    def __call__(self, pairwise_cosines) -> typing.Union[torch.Tensor, np.ndarray]:
        if self.pair_only:
            comp = pairwise_cosines[:, self.correct_index] > pairwise_cosines[:, self.pair_comparison_index]
        else:
            comp = pairwise_cosines.argmax(dim=1) == self.correct_index  # type: ignore

        return comp.to(torch.float)
        
        
class DifferenceMetric(Metric):
    combine_func: typing.Callable
    n_stimuli: int

    def __init__(self, name: str, correct_index: int = 0, combine_func: typing.Callable = torch.mean,
                 combine_func_kwargs: typing.Dict[str, typing.Any] = dict(dim=1), n_stimuli: int = 3):
        super(DifferenceMetric, self).__init__(name, correct_index)
        self.combine_func = combine_func
        self.combine_func_kwargs = combine_func_kwargs
        self.n_stimuli = n_stimuli

        self.incorrect_indices = list(range(self.n_stimuli))
        self.incorrect_indices.remove(correct_index)
        
    def __call__(self, pairwise_cosines) -> typing.Union[torch.Tensor, np.ndarray]:
        return pairwise_cosines[:, self.correct_index] - self.combine_func(pairwise_cosines[:, self.incorrect_indices], **self.combine_func_kwargs)
    
    
class PairDifferenceMetric(Metric):
    def __init__(self, name, correct_index=0, pair_comparison_index=1):
        super(PairDifferenceMetric, self).__init__(name, correct_index)
        self.pair_comparison_index = pair_comparison_index
        
    def __call__(self, pairwise_cosines) -> typing.Union[torch.Tensor, np.ndarray]:
        return pairwise_cosines[:, self.correct_index] - pairwise_cosines[:, self.pair_comparison_index]
    
    
METRICS = (AccuracyMetric('Accuracy', pair_only=True), 
#            AccuracyMetric('Accuracy(Triplet)'),
           PairDifferenceMetric('Difference'),
#            DifferenceMetric('MeanDiff'),
#            DifferenceMetric('MaxDiff', combine_func=lambda x: torch.max(x, dim=1).values, combine_func_kwargs={})
          )

TaskResults = namedtuple('TaskResults', ('mean', 'std', 'n'))

BATCH_SIZE = 32


def quinn_embedding_task_single_generator(
    model, triplet_generator, metrics=METRICS, N=1024, batch_size=BATCH_SIZE, use_tqdm=False, device=None, tsne_mode=False):
    
    if device is None:
        device = next(model.parameters()).device

    if tsne_mode and N != 1:
        raise ValueError(f'TSNE mode only works with N=1, received N={N}')

    data = triplet_generator(N)
    if tsne_mode:
        data = data.squeeze(0)
        output_embeddings = []
    
    B = batch_size
    dataloader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=False)
    
    model_results = defaultdict(list)
    cos = nn.CosineSimilarity(dim=-1)
    triangle_indices = np.triu_indices(3, 1)
    
    model.eval()
    
    data_iter = dataloader
    if use_tqdm:
        data_iter = tqdm(dataloader, desc='Batches')
        
    for b in data_iter:
        if tsne_mode:
            x = b[0]  # shape (B, 3, 224, 224) where H is the number of habituation stimuli
            x = x.to(device)
            e = model(x).detach().cpu().numpy()
            output_embeddings.append(e)

        else:
            x = b[0]  # shape (B, H + 2, 3, 224, 224) where H is the number of habituation stimuli
            H = x.shape[1] - 2
            x = x.view(-1, *x.shape[2:])
            x = x.to(device)
            e = model(x).detach()
            e = e.view(B, H + 2, -1)  # shape (B, H + 2, Z)
            
            if H > 1:  # if we have multiple habituation stimuli, average them
                average_habituation_embedding = e[:, :-2, :].mean(dim=1, keepdim=True)
                test_embeddings = e[:, -2:]
                e = torch.cat((average_habituation_embedding, test_embeddings), dim=1)

            embedding_pairwise_cosine = cos(e[:, :, None, :], e[:, None, :, :])  # shape (B, 3, 3)
            triplet_cosines = embedding_pairwise_cosine[:, triangle_indices[0], triangle_indices[1]] # shape (B, 3)

            # triplet_cosines

            for metric in metrics:
                model_results[metric.name].append(metric(triplet_cosines).cpu())

    if not tsne_mode:
        for metric in metrics:
            model_results[metric.name] = metric.aggregate(model_results[metric.name])

    del dataloader

    if tsne_mode:
        return data.cpu().numpy(), np.concatenate(output_embeddings, axis=0)

    del data
    
    return model_results


def quinn_embedding_task_multiple_generators(
    model, condition_names, triplet_generators, 
    metrics=METRICS, N=1024, batch_size=BATCH_SIZE, tsne_mode=False):
    
    all_results = {}
    for condition_name, triplet_gen in zip(condition_names, triplet_generators):
        results = quinn_embedding_task_single_generator(model, triplet_gen, metrics=metrics,
                                                        N=N, batch_size=batch_size, tsne_mode=tsne_mode)

        if tsne_mode:
            data, results = results
            all_results[condition_name] = dict(data=data, results=results)
        else:                                                    
            all_results[condition_name] = {metric.name: TaskResults(np.mean(results[metric.name]), 
                                                        np.std(results[metric.name]), N) 
                                           for metric in metrics}
    return all_results


def run_multiple_models_multiple_generators(model_names, model_kwarg_dicts, 
                                            condition_names, condition_generators, N, 
                                            batch_size=BATCH_SIZE, tsne_mode=False):
    all_model_results = {}
    
    for name, model_kwargs in zip (model_names, model_kwarg_dicts):
        print(f'Starting model {name}')
        model = build_model(**model_kwargs)

        all_model_results[name] = quinn_embedding_task_multiple_generators(
            model, condition_names, condition_generators, N=N, batch_size=batch_size, tsne_mode=tsne_mode)

        del model

        torch.cuda.empty_cache()
        
    return all_model_results

