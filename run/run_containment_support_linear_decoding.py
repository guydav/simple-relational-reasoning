
import argparse
import copy
import cProfile
import itertools
import matplotlib
import os
import sys
from tqdm import tqdm, trange
import typing

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

from simple_relational_reasoning.embeddings.containment_support_dataset import ContainmentSupportDataset

import numpy as np
import pandas as pd
import torch

from simple_relational_reasoning.embeddings.models import MODELS, RESNEXT, FLIPPING_OPTIONS, DINO_OPTIONS
from simple_relational_reasoning.embeddings.containment_support_dataset import DEFAULT_VALIDATION_PROPORTION
from simple_relational_reasoning.embeddings.containment_support_linear_decoding import run_containment_support_linear_decoding_multiple_models, BATCH_SIZE, DEFAULT_PATIENCE_EPOCHS, DEFAULT_PATIENCE_MARGIN
from simple_relational_reasoning.embeddings.tables import multiple_results_to_df

matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['figure.edgecolor'] = (1, 1, 1, 0)
matplotlib.rcParams['figure.facecolor'] = (1, 1, 1, 0)

parser = argparse.ArgumentParser()

DEFAULT_SEED = 33
parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed to run with')

DEFAULT_DATASET_PATH = '/home/gd1279/scratch/containment_support/containment_both_new_baskets'
parser.add_argument('-d', '--dataset-path', type=str, default=DEFAULT_DATASET_PATH, help='Path to dataset')

parser.add_argument('-n', '--n-epochs', type=int, default=100, help='Number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--validation-proportion', type=float, default=DEFAULT_VALIDATION_PROPORTION, help='Proportion of dataset to use for validation')
parser.add_argument('--patience-epochs', type=int, default=DEFAULT_PATIENCE_EPOCHS, help='# Epochs to use for patience')
parser.add_argument('--patience-margin', type=float, default=DEFAULT_PATIENCE_MARGIN, help='Improvement margin to use for patience')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--by-target-object', action='store_true', help='Whether to test by target object')
group.add_argument('--by-reference-object', action='store_true', help='Whether to test by reference object')
group.add_argument('--test-proportion', type=float, default=None, help='Proportion of dataset to use for testing')

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

    dataset = ContainmentSupportDataset(args.dataset_path)

    all_model_results = run_containment_support_linear_decoding_multiple_models(
        model_names, model_kwarg_dicts, dataset, 
        args.n_epochs, args.lr, args.by_target_object, args.by_reference_object, args.test_proportion,
        args.batch_size, args.validation_proportion, args.patience_epochs, args.patience_margin, args.seed)

    return pd.DataFrame.from_records(all_model_results)


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

    out_df = handle_single_args_setting(main_args)
    # out_df = pd.concat(dataframes)
    # out_df.reset_index(drop=True, inplace=True)
    output_file = main_args.output_file

    output_folder, _ = os.path.split(output_file)
    os.makedirs(output_folder, exist_ok=True)

    while os.path.exists(output_file):
        output_file += '_1'
    out_df.to_csv(output_file)
