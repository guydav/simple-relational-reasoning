
import argparse
import copy
import cProfile
import itertools
import matplotlib
import os
import sys
from tqdm import tqdm, trange

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
import torch

from simple_relational_reasoning.embeddings.stimuli import STIMULUS_GENERATORS
from simple_relational_reasoning.embeddings.triplets import TRIPLET_GENERATORS, DEFAULT_MULTIPLE_HABITUATION_RADIUS, DEFAULT_MARGIN_BUFFER
from simple_relational_reasoning.embeddings.models import MODELS, FLIPPING_OPTIONS, DINO_OPTIONS, RESNEXT
from simple_relational_reasoning.embeddings.task import run_multiple_models_multiple_generators, BATCH_SIZE
from simple_relational_reasoning.embeddings.tables import multiple_results_to_df

matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['figure.edgecolor'] = (1, 1, 1, 0)
matplotlib.rcParams['figure.facecolor'] = (1, 1, 1, 0)

parser = argparse.ArgumentParser()

DEFAULT_SEED = 33
parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed to run with')

parser.add_argument('--replications', type=int, default=1, help='Number of replications to run')

DEFAULT_N = 1024
parser.add_argument('-n', '--n-examples', type=int, default=DEFAULT_N, help='Number of examples to generate')

parser.add_argument('-t', '--triplet-generator', type=str, 
    choices=list(TRIPLET_GENERATORS.keys()), help='Which triplet generator to run with')

parser.add_argument('-s', '--stimulus-generators', action='append', required=True,
    choices=list(STIMULUS_GENERATORS.keys()), help='Stimulus generator to run with')

parser.add_argument('--stimulus-generator-kwargs', action='append', default=list(),
    help='Specify key=value pairs to pass to the stimulus generator.')

DEFAULT_N_TARGET_TYPES = None
VALID_N_TARGET_TYPES = list(range(1, 4))
parser.add_argument('--n-target-types', type=int, default=DEFAULT_N_TARGET_TYPES, choices=VALID_N_TARGET_TYPES)

parser.add_argument('--n-habituation-stimuli', type=int, default=None, help='Number of habituation stimuli')
parser.add_argument('--multiple-habituation-radius', type=int, default=DEFAULT_MULTIPLE_HABITUATION_RADIUS, 
    help='Radius to place multiple habituation stimuli in')

parser.add_argument('--margin-buffer', type=int, default=DEFAULT_MARGIN_BUFFER, help='Buffer to add to the margin')

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
parser.add_argument('--unpooled-output', action='store_true', help='Use unpooled model outputs')

parser.add_argument('-o', '--output-file', type=str, help='Output file to write to')

parser.add_argument('--tqdm', action='store_true', help='Use tqdm progress bar')

parser.add_argument('--device', default=None, help='Which device to use. Defaults to cuda:0 if available and cpu if not.')
parser.add_argument('-b', '--batch-size', type=int, default=BATCH_SIZE, help='Batch size to use')

parser.add_argument('--profile', action='store_true', help='Profile')
parser.add_argument('--profile-output', default='/home/gd1279/scratch/profile/embeddings_profile')

parser.add_argument('--memory-profile', action='store_true', help='Profile memory usage')

parser.add_argument('--print-setting-options', action='store_true')

MULTIPLE_OPTION_FIELD_DEFAULTS = {
    'n_target_types': [1, 2],
    'n_habituation_stimuli': [1, 4],
}
MULTIPLE_OPTION_REWRITE_FIELDS = list(MULTIPLE_OPTION_FIELD_DEFAULTS.keys())

SINGLE_OPTION_FIELDS_TO_DF = ['seed', 'n_examples', 'unpooled_output']


def create_triplet_generators(args):
    triplet_generator_class = TRIPLET_GENERATORS[args.triplet_generator]

    triplet_generators = []

    for stimulus_generator_name in args.stimulus_generators:
        stimulus_generator_builder = STIMULUS_GENERATORS[stimulus_generator_name]        
        stimulus_generator = stimulus_generator_builder(**args.stimulus_generator_kwargs)
        triplet_generator_kwargs = dict(n_target_types=args.n_target_types,
            margin_buffer=args.margin_buffer,
            n_habituation_stimuli=args.n_habituation_stimuli, 
            multiple_habituation_radius=args.multiple_habituation_radius)
        
        if 'same_horizontal_half' in vars(args):
            triplet_generator_kwargs['same_horizontal_half'] = args.same_horizontal_half

        triplet_generator = triplet_generator_class(stimulus_generator, **triplet_generator_kwargs)
        triplet_generators.append(triplet_generator)

    return triplet_generators
    

def handle_multiple_option_defaults(args):
    if args.triplet_generator == 'same_half':
        MULTIPLE_OPTION_REWRITE_FIELDS.append('same_horizontal_half')
        MULTIPLE_OPTION_FIELD_DEFAULTS['same_horizontal_half'] = [True, False]

    var_args = vars(args)
    for key in MULTIPLE_OPTION_FIELD_DEFAULTS:
        if key not in var_args or var_args[key] is None or (hasattr(var_args[key], '__len__') and len(var_args[key]) == 0):
            var_args[key] = MULTIPLE_OPTION_FIELD_DEFAULTS[key]

        elif not hasattr(var_args[key], '__len__'):
            var_args[key] = [var_args[key]]

    return args


def handle_single_args_setting(args):
    var_args = vars(args)

    if args.print_setting_options:
        print(' ' * 26 + 'Setting Options')
        for k in MULTIPLE_OPTION_REWRITE_FIELDS:
            print(' ' * 26 + k + ': ' + str(var_args[k]))
    
    model_kwarg_dicts = []
    model_names = []
    for model_name in args.model:
        if args.saycam:
            model_kwarg_dicts.append(dict(name=model_name, device=args.device, pretrained=False, saycam=args.saycam, unpooled_output=args.unpooled_output))
            model_names.append(f'{model_name}-saycam({args.saycam})')
        
        if args.imagenet:
            model_kwarg_dicts.append(dict(name=model_name, device=args.device, pretrained=True, unpooled_output=args.unpooled_output))
            model_names.append(f'{model_name}-imagenet')

        if args.untrained:
            model_kwarg_dicts.append(dict(name=model_name, device=args.device, pretrained=False, unpooled_output=args.unpooled_output))
            model_names.append(f'{model_name}-random')

        if model_name == RESNEXT and args.flipping and len(args.flipping) > 0:
            for flip_type in args.flipping:
                model_kwarg_dicts.append(dict(name=model_name, device=args.device, 
                    pretrained=False, flip=flip_type, unpooled_output=args.unpooled_output))

                model_names.append(f'{model_name}-saycam(S)-{flip_type}')

        if model_name == RESNEXT and args.dino and len(args.dino) > 0:
            for dino in args.dino:
                model_kwarg_dicts.append(dict(name=model_name, device=args.device, 
                    pretrained=False, dino=dino, unpooled_output=args.unpooled_output))

                model_names.append(f'{model_name}-DINO-{dino}')

    var_args['stimulus_generator_kwargs'] = {s.split('=')[0]: s.split('=')[1] for s in args.stimulus_generator_kwargs}

    all_model_results = []

    rep_iter = trange(args.replications) if (args.tqdm and args.replications > 1) else range(args.replications)
    for r in rep_iter:
        var_args['seed'] = args.seed + 1
        torch.manual_seed(args.seed)
        var_args['stimulus_generator_kwargs']['rng'] = np.random.default_rng(args.seed)

        triplet_generators = create_triplet_generators(args)

        all_model_results.append(run_multiple_models_multiple_generators(
            model_names, model_kwarg_dicts, args.stimulus_generators, 
            triplet_generators, args.n_examples, args.batch_size))

        del triplet_generators


    result_df = multiple_results_to_df(all_model_results, N=args.n_examples)

    for key in MULTIPLE_OPTION_REWRITE_FIELDS + SINGLE_OPTION_FIELDS_TO_DF:
        result_df[key] = var_args[key]

    result_df['relation'] = 'no_reference'
    result_df['triplet_generator'] = args.triplet_generator

    return result_df


if __name__ == '__main__':
    main_args = parser.parse_args()
    main_args = handle_multiple_option_defaults(main_args)
    main_var_args = vars(main_args)

    heap = None
    if main_args.memory_profile:
        heap = hpy()
        heap.setrelheap()

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

    multiple_option_field_values = [main_var_args[key] for key in MULTIPLE_OPTION_REWRITE_FIELDS]

    dataframes = []

    value_iter = itertools.product(*multiple_option_field_values)
    if main_args.tqdm:
        total = np.prod([len(v) for v in multiple_option_field_values])
        value_iter = tqdm(value_iter, desc='Setting', total=total)

    try:
        for i, value_combination in enumerate(value_iter):
            args_copy = copy.deepcopy(main_args)
            var_args_copy = vars(args_copy)
            var_args_copy.update({key: value for key, value in zip(MULTIPLE_OPTION_REWRITE_FIELDS,
                                                                value_combination)})

            if main_args.profile:
                print('Profiling...')
                cProfile.run('handle_single_args_setting(args_copy)', f'{main_args.profile_output}_{i}')
            else:
                dataframes.append(handle_single_args_setting(args_copy))

    finally:
        if len(dataframes) > 0:
            out_df = pd.concat(dataframes)
            out_df.reset_index(drop=True, inplace=True)
            output_file = main_args.output_file

            output_folder, _ = os.path.split(output_file)
            os.makedirs(output_folder, exist_ok=True)

            while os.path.exists(output_file):
                output_file += '_1'
            out_df.to_csv(output_file)
