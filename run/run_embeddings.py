
import argparse
import copy
import itertools
import os
import sys

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
import torch

from simple_relational_reasoning.embeddings.stimuli import STIMULUS_GENERATORS
from simple_relational_reasoning.embeddings.triplets import TRIPLET_GENERATORS, RELATIONS, DEFAULT_MULTIPLE_HABITUATION_RADIUS
from simple_relational_reasoning.embeddings.models import MODELS
from simple_relational_reasoning.embeddings.task import run_multiple_models_multiple_generators
from simple_relational_reasoning.embeddings.tables import table_per_relation_multiple_results


parser = argparse.ArgumentParser()

DEFAULT_SEED = 33
parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed to run with')

parser.add_argument('--replications', type=int, default=1, help='Number of replications to run')

DEFAULT_N = 1024
parser.add_argument('-n', '--n-examples', type=int, default=DEFAULT_N, help='Number of examples to generate')

parser.add_argument('-s', '--stimulus-generator', type=str, 
    choices=list(STIMULUS_GENERATORS.keys()), help='Stimulu generator to run with')

parser.add_argument('--stimulus-generator-kwargs', action='append', 
    Help='Specify key=value pairs to pass to the stimulus generator.')

DEFAULT_DISTANCE_ENDPOINTS = (30, 70)
parser.add_argument('--target-distance-endpoints', type=int, nargs=2, default=DEFAULT_DISTANCE_ENDPOINTS,)


parser.add_argument('-r', '--relation', type=str, action='append', choices=RELATIONS,
                    help='Which relation(s) to run (default: all)')

DEFAULT_TWO_REFERENCE_OBJECTS = None
parser.add_argument('--two-reference-objects', type=int, default=DEFAULT_TWO_REFERENCE_OBJECTS)

DEFAULT_TRANSPOSE = None
parser.add_argument('--transpose-stimuli', type=int, default=DEFAULT_TRANSPOSE)

DEFAULT_N_TARGET_TYPES = None
VALID_N_TARGET_TYPES = list(range(1, 4))
parser.add_argument('--n-target-types', type=int, default=DEFAULT_N_TARGET_TYPES, choices=VALID_N_TARGET_TYPES)

parser.add_argument('-h', '--n-habituation_stimuli', type=int, default=1, help='Number of habituation stimuli')

parser.add_argument('--multiple-habituation-radius', type=int, default=DEFAULT_MULTIPLE_HABITUATION_RADIUS, 
    help='Radius to place multiple habituation stimuli in')

parser.add_argument('-t', '--triplet-generator', type=str, 
    choices=list(TRIPLET_GENERATORS.keys()), help='Which triplet generator to run with')

parser.add_argument('--base-model-name', type=str, default='', help='Base name for the models')

parser.add_argument('-m', '--model', type=str, action='append', choices=MODELS,
                    help='Which models to run')

parser.add_argument('--saycam', type='str', default=None, help='Which SAYcam model to use')
parser.add_argument('--imagenet-pretrained', action='store_true', help='Use imagenet pretrained models')
parser.add_argument('--untrained', action='store_true', help='Use untrained models')

parser.add_argument('-o', '--output-file', type=str, help='Output file to write to')

parser.add_argument('--rotate-angle', type=int, default=None, help='Angle to rotate the stimuli by')


MULTIPLE_OPTION_FIELD_DEFAULTS = {
    'relation': RELATIONS,
    'two_reference_objects': [0, 1],
    'transpose_stimuli': [0, 1],
    'n_target_types': VALID_N_TARGET_TYPES,
    'n_habituation_stimuli': [1, 4,],
    'rotate_angle': [-45, -30, 0, 30, 45],
}
MULTIPLE_OPTION_REWRITE_FIELDS = list(MULTIPLE_OPTION_FIELD_DEFAULTS.keys())

def default_name_func(generator_kwargs, base_name=''):
    two_reference_objects = generator_kwargs['two_reference_objects']
    transpose = generator_kwargs['transpose']
    
    if two_reference_objects:
        if transpose:
            rel_name = 'VerticalBetween'
        else:
            rel_name = 'Between'
    else:
        if transpose:
            rel_name = 'Left/Right'
        else:
            rel_name = 'Above/Below'
    
    name = f'{base_name}{base_name and "-" or ""}{rel_name}'
            
    n_target_types = None
    if 'n_target_types' in generator_kwargs:
        n_target_types = generator_kwargs['n_target_types']
        return name + f'-{n_target_types}-types'
    
    return name


def create_generators_and_names(triplet_generator_class, stimulus_generator, kwarg_value_sets, 
                                name_func=default_name_func, name_func_kwargs=None):
    if name_func_kwargs is None:
        name_func_kwargs = {}
    
    names = []
    triplet_generators = []

    for value_set in itertools.product(*kwarg_value_sets.values()):
        kwargs = {k: v for (k, v) in zip(kwarg_value_sets.keys(), value_set)}
        names.append(name_func(kwargs, **name_func_kwargs))
        triplet_generators.append(triplet_generator_class(stimulus_generator, **kwargs))

    return names, triplet_generators
    

def handle_multiple_option_defaults(args):
    var_args = vars(args)
    for key in MULTIPLE_OPTION_FIELD_DEFAULTS:
        if key not in var_args or var_args[key] is None or (hasattr(var_args[key], '__len__') and len(var_args[key]) == 0):
            var_args[key] = MULTIPLE_OPTION_FIELD_DEFAULTS[key]

        elif not hasattr(var_args[key], '__len__'):
            var_args[key] = [var_args[key]]

    return args


def handle_single_args_setting(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'
    
    model_kwarg_dicts = []
    for model_name in args.model:
        if args.saycam:
            model_kwarg_dicts.append(name=model_name, device=device, pretrained=False, saycam=args.saycam)
        
        if args.imagenet_pretrained:
            model_kwarg_dicts.append(name=model_name, device=device, pretrained=True)

        if args.untrained:
            model_kwarg_dicts.append(name=model_name, device=device, pretrained=False)

    model_names = [ f'{d["name"]}-{"saycam({s})".format(s=d["syacam"]) if "saycam" in d else (d["pretrained"] and "imagenet" or "random")}'
               for d in model_kwarg_dicts]

    stimulus_generator_builder = STIMULUS_GENERATORS[args.stimulus_generator]
    stimulus_generator_kwargs = {s.split('=')[0]: s.split('=')[1] for s in args.stimulus_generator_kwargs}

    all_model_results = []
    for r in range(args.n_replications):
        stimulus_generator_kwargs['rng'] = np.random.default_rng(args.seed + r)
        stimulus_generator_kwargs['rotate_angle'] = args.rotate_angle
        stimulus_generator = stimulus_generator_builder(**stimulus_generator_kwargs) 

        value_sets = {key: getattr(args, key) if getattr(args, key) is not None else default_value
            for key, default_value in MULTIPLE_OPTION_FIELD_DEFAULTS.items()}

        triplet_names, triplet_generators = create_generators_and_names(
            TRIPLET_GENERATORS[args.triplet_generator], stimulus_generator, 
            value_sets, name_func_kwargs=dict(base_name=args.base_model_name))

        all_model_results.append(run_multiple_models_multiple_generators(
            model_names, model_kwarg_dicts, triplet_names, triplet_generators, args.n_examples))

    if len(all_model_results) == 1:
        all_model_results = all_model_results[0]

    result_df = table_per_relation_multiple_results(all_model_results, N=args.n_examples,
        display_tables=False)

    for multiple_option_key in MULTIPLE_OPTION_REWRITE_FIELDS:
        result_df[multiple_option_key] = args.__dict__[multiple_option_key]

    result_df.to_csv(args.output_file)


if __name__ == '__main__':
    main_args = parser.parse_args()
    main_var_args = vars(main_args)
    print(' ' * 26 + 'Global Options')
    for k, v in main_var_args.items():
        print(' ' * 26 + k + ': ' + str(v))

    multiple_option_field_values = [main_var_args[key] for key in MULTIPLE_OPTION_REWRITE_FIELDS]

    dataframes = []

    for value_combination in itertools.product(*multiple_option_field_values):
        args_copy = copy.deepcopy(main_args)
        var_args_copy = vars(args_copy)
        var_args_copy.update({key: value for key, value in zip(MULTIPLE_OPTION_REWRITE_FIELDS,
                                                               value_combination)})
    
        # TODO: any checks for arg combinations we shouldn't run?
        dataframes.append(handle_single_args_setting(args_copy))

    out_df = pd.concat(dataframes)
    out_df.reset_index(drop=True, inplace=True)
    out_df.to_csv(main_args.output_file)
