
import argparse
import copy
import gzip
import itertools
import matplotlib
import os
import sys
from tqdm import tqdm, trange
import typing
import pickle

import numpy as np
import pandas as pd
import torch


sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

from simple_relational_reasoning.embeddings.models import MODELS, RESNEXT, FLIPPING_OPTIONS, DINO_OPTIONS
from simple_relational_reasoning.embeddings.containment_support_task import run_containment_support_task_multiple_models, BATCH_SIZE
from simple_relational_reasoning.embeddings.tables import multiple_results_to_df
from simple_relational_reasoning.embeddings.containment_support_dataset import QUINN_SCENE_TYPES, SCENE_TYPES, ContainmentSupportDataset


matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['figure.edgecolor'] = (1, 1, 1, 0)
matplotlib.rcParams['figure.facecolor'] = (1, 1, 1, 0)

parser = argparse.ArgumentParser()

DEFAULT_SEED = 33
parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed to run with')

DEFAULT_DATASET_PATH = '/home/gd1279/scratch/containment_support/containment_both_new_baskets'
parser.add_argument('-d', '--dataset-path', type=str, default=DEFAULT_DATASET_PATH, help='Path to dataset')

parser.add_argument('--shuffle-habituation-stimuli', action='store_true', help='Shuffle habituation stimuli')
parser.add_argument('--quinn-stimuli', action='store_true', help='Run using Quinn stimuli')
parser.add_argument('--tsne', action='store_true', help='Run t-SNE on embeddings')

parser.add_argument('--aggregate-results', action='store_true', help='Aggregate results')

# TODO: add flags for separating the results by bowl (reference) color, or for running with different target objects

parser.add_argument('--base-model-name', type=str, default='', help='Base name for the models')

parser.add_argument('-m', '--model', type=str, action='append', choices=MODELS,
                    help='Which models to run')

parser.add_argument('--saycam', type=str, default=None, help='Which SAYcam model to use')
parser.add_argument('--imagenet', action='store_true', help='Use imagenet pretrained models')
parser.add_argument('--untrained', action='store_true', help='Use untrained models')
parser.add_argument('--flipping', action='append',
    choices=FLIPPING_OPTIONS, help='Use one of the flipping models Emin created')
parser.add_argument('--dino', action='append',
    choices=DINO_OPTIONS, help='Use one of the DINO models Emin created')

parser.add_argument('-o', '--output-file', type=str, help='Output file to write to')

parser.add_argument('--tqdm', action='store_true', help='Use tqdm progress bar')

parser.add_argument('--device', default=None, help='Which device to use. Defaults to cuda:0 if available and cpu if not.')
parser.add_argument('-b', '--batch-size', type=int, default=BATCH_SIZE, help='Batch size to use')

    
# def handle_multiple_option_defaults(args):
#     var_args = vars(args)
#     for key in MULTIPLE_OPTION_FIELD_DEFAULTS:
#         if key not in var_args or var_args[key] is None or (hasattr(var_args[key], '__len__') and len(var_args[key]) == 0):
#             var_args[key] = MULTIPLE_OPTION_FIELD_DEFAULTS[key]

#         elif not hasattr(var_args[key], '__len__'):
#             var_args[key] = [var_args[key]]

#     return args


def handle_single_args_setting(args):    
    model_kwarg_dicts = []
    model_names = []
    for model_name in args.model:
        if args.saycam:
            model_kwarg_dicts.append(dict(name=model_name, device=args.device, pretrained=False, saycam=args.saycam))
            model_names.append(f'{model_name}-saycam({args.saycam})')
        
        if args.imagenet:
            model_kwarg_dicts.append(dict(name=model_name, device=args.device, pretrained=True))
            model_names.append(f'{model_name}-imagenet')

        if args.untrained:
            model_kwarg_dicts.append(dict(name=model_name, device=args.device, pretrained=False))
            model_names.append(f'{model_name}-random')

        if model_name == RESNEXT and args.flipping and len(args.flipping) > 0:
            for flip_type in args.flipping:
                model_kwarg_dicts.append(dict(name=model_name, device=args.device, 
                    pretrained=False, flip=flip_type))

                model_names.append(f'{model_name}-saycam(S)-{flip_type}')

        if model_name == RESNEXT and args.dino and len(args.dino) > 0:
            for dino in args.dino:
                model_kwarg_dicts.append(dict(name=model_name, device=args.device, 
                    pretrained=False, dino=dino))

                model_names.append(f'{model_name}-DINO-{dino}')

    dataset = ContainmentSupportDataset(args.dataset_path, shuffle_habituation_stimuli=args.shuffle_habituation_stimuli,
        scene_types=QUINN_SCENE_TYPES if args.quinn_stimuli else SCENE_TYPES, random_seed=args.seed)

    all_model_results = run_containment_support_task_multiple_models(
        model_names, model_kwarg_dicts, dataset, batch_size=args.batch_size, 
        tsne_mode=args.tsne, aggregate_results=args.aggregate_results)

    output_file = validate_output_path(args)

    if args.tsne:
        with gzip.open(output_file, 'wb') as f:
            pickle.dump(all_model_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    elif not args.aggregate_results:
        result_df = containment_support_results_to_df(all_model_results, dataset)
        result_df.to_csv(output_file)

    else:
        raise NotImplementedError('Aggregate results not implemented yet')


def containment_support_results_to_df(all_model_results: typing.Dict[str, np.ndarray], dataset: ContainmentSupportDataset):
    df_rows = []
    for model_name, model_results in all_model_results.items():
        for example_index in range(model_results.shape[0]):
            df_rows.append([model_name, example_index, 
                dataset.dataset_configuration_indices[example_index], 
                dataset.dataset_reference_objects[example_index],
                dataset.dataset_target_objects[example_index],
                dataset.dataset_habituation_target_objects[example_index],
                *model_results[example_index]])

    headers = ['model', 'example_index', 'configuration_index', 'reference_object', 'target_object'] + [f'{t1}_{t2}_cos' for t1, t2 in itertools.combinations(dataset.scene_types, 2)]
    return pd.DataFrame(df_rows, columns=headers)


def validate_output_path(args):
    output_file = args.output_file

    output_folder, _ = os.path.split(output_file)
    os.makedirs(output_folder, exist_ok=True)

    while os.path.exists(output_file):
        output_file += '_1'
    return output_file


if __name__ == '__main__':
    main_args = parser.parse_args()
    main_var_args = vars(main_args)

    if main_args.device is not None:
        main_args.device = torch.device(main_args.device)

    else:
        if torch.cuda.is_available():
            main_args.device = torch.device('cuda:0')
        else:
            main_args.device = 'cpu'

    print(' ' * 26 + 'Global Options')
    for k, v in main_var_args.items():
        print(' ' * 26 + k + ': ' + str(v))

    handle_single_args_setting(main_args)
    # out_df = pd.concat(dataframes)
    # out_df.reset_index(drop=True, inplace=True)
    


