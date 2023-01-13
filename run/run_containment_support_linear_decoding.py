
import argparse
import copy
import cProfile
import itertools
import matplotlib
import os
import sys
from tqdm import tqdm, trange
import typing

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

from run_utils import args_to_model_configurations
from simple_relational_reasoning.embeddings.containment_support_dataset import ContainmentSupportDataset
from simple_relational_reasoning.embeddings.models import ALL_MODELS, RESNEXT, FLIPPING_OPTIONS, DINO_OPTIONS
from simple_relational_reasoning.embeddings.containment_support_dataset import DEFAULT_VALIDATION_PROPORTION
from simple_relational_reasoning.embeddings.containment_support_linear_decoding import run_containment_support_linear_decoding_multiple_models, \
    BATCH_SIZE, DEFAULT_PATIENCE_EPOCHS, DEFAULT_PATIENCE_MARGIN, DEFAULT_N_TEST_PROPORTION_RANDOM_SEEDS
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

parser.add_argument('--by-target-object', action='store_true', help='Whether to test by target object')
parser.add_argument('--by-reference-object', action='store_true', help='Whether to test by reference object')
parser.add_argument('--test-proportion', type=float, default=None, help='Proportion of dataset to use for testing')

parser.add_argument('--n-test-proportion-random-seeds', type=int, default=DEFAULT_N_TEST_PROPORTION_RANDOM_SEEDS, help='Number of processes to use')
parser.add_argument('--base-model-name', type=str, default='', help='Base name for the models')

parser.add_argument('-m', '--model', type=str, action='append', choices=ALL_MODELS,
                    help='Which models to run')

parser.add_argument('--saycam', type=str, default=None, help='Which SAYcam model to use')
parser.add_argument('--imagenet', action='store_true', help='Use imagenet pretrained models')
parser.add_argument('--untrained', action='store_true', help='Use untrained models')
parser.add_argument('--flipping', action='append',
    choices=FLIPPING_OPTIONS, help='Use one of the flipping models Emin created')
parser.add_argument('--dino', action='append',
    choices=DINO_OPTIONS, help='Use one of the DINO models Emin created')
parser.add_argument('--unpooled-output', action='store_true', help='Use unpooled model outputs')

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
    model_kwarg_dicts, model_names = args_to_model_configurations(args)

    dataset = ContainmentSupportDataset(args.dataset_path)

    all_model_results, all_model_per_example_results = run_containment_support_linear_decoding_multiple_models(
        model_names, model_kwarg_dicts, dataset, 
        args.n_epochs, args.lr, args.by_target_object, args.by_reference_object, args.test_proportion, args.n_test_proportion_random_seeds,
        args.batch_size, args.validation_proportion, args.patience_epochs, args.patience_margin, args.seed)

    result_df = pd.DataFrame.from_records(all_model_results)
    result_df = result_df.assign(global_seed=args.seed, unpooled_output=args.unpooled_output)

    per_example_df = pd.DataFrame.from_records(all_model_per_example_results)
    per_example_df = per_example_df.assign(global_seed=args.seed, unpooled_output=args.unpooled_output)

    return result_df, per_example_df


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

    out_df, out_per_example_df = handle_single_args_setting(main_args)
    output_file = main_args.output_file

    output_folder, _ = os.path.split(output_file)
    os.makedirs(output_folder, exist_ok=True)

    while os.path.exists(output_file):
        output_file += '_1'
    out_df.to_csv(output_file)

    per_example_output_file = output_file.replace('.csv', '_per_example.csv')
    out_per_example_df.to_csv(per_example_output_file)

