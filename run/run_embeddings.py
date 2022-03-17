
import argparse
import copy
import cProfile
import itertools
import os
import sys
from tqdm import tqdm, trange

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
import torch

from simple_relational_reasoning.embeddings.stimuli import STIMULUS_GENERATORS
from simple_relational_reasoning.embeddings.triplets import TRIPLET_GENERATORS, RELATIONS, ABOVE_BELOW_RELATION, BETWEEN_RELATION, DEFAULT_MULTIPLE_HABITUATION_RADIUS
from simple_relational_reasoning.embeddings.models import MODELS
from simple_relational_reasoning.embeddings.task import run_multiple_models_multiple_generators
from simple_relational_reasoning.embeddings.tables import multiple_results_to_df


parser = argparse.ArgumentParser()

DEFAULT_SEED = 33
parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed to run with')

parser.add_argument('--replications', type=int, default=1, help='Number of replications to run')

DEFAULT_N = 1024
parser.add_argument('-n', '--n-examples', type=int, default=DEFAULT_N, help='Number of examples to generate')

parser.add_argument('-s', '--stimulus-generators', action='append', required=True,
    choices=list(STIMULUS_GENERATORS.keys()), help='Stimulus generator to run with')

parser.add_argument('--stimulus-generator-kwargs', action='append', default=list(),
    help='Specify key=value pairs to pass to the stimulus generator.')

DEFAULT_DISTANCE_ENDPOINTS = (-1, -1)
DISTANCE_ENDPOINTS_DICT = {  # two_reference_objects, adjacent_reference_objects
    (False, False): (30, 80),
    (True, False): (50, 80),
    (True, True): (40, 80),  
}
parser.add_argument('--distance-endpoints', type=int, nargs=2, default=DEFAULT_DISTANCE_ENDPOINTS,)

parser.add_argument('-r', '--relation', type=str, action='append', choices=RELATIONS,
                    help='Which relation(s) to run (default: all)')

DEFAULT_TWO_REFERENCE_OBJECTS = None
parser.add_argument('--two-reference-objects', type=int, default=DEFAULT_TWO_REFERENCE_OBJECTS)

DEFAULT_ADJACENT_REFERENCE_OBJECTS = None
parser.add_argument('--adjacent-reference-objects', type=int, default=DEFAULT_ADJACENT_REFERENCE_OBJECTS)

DEFAULT_TRANSPOSE = None
parser.add_argument('--transpose-stimuli', type=int, default=DEFAULT_TRANSPOSE)

DEFAULT_N_TARGET_TYPES = None
VALID_N_TARGET_TYPES = list(range(1, 4))
parser.add_argument('--n-target-types', type=int, default=DEFAULT_N_TARGET_TYPES, choices=VALID_N_TARGET_TYPES)

parser.add_argument('--n-habituation_stimuli', type=int, default=None, help='Number of habituation stimuli')

parser.add_argument('--multiple-habituation-radius', type=int, default=DEFAULT_MULTIPLE_HABITUATION_RADIUS, 
    help='Radius to place multiple habituation stimuli in')

parser.add_argument('--extra-diagonal-margin', type=int, default=0, help='Extra margin to add to diagonal stimuli')

parser.add_argument('-t', '--triplet-generator', type=str, 
    choices=list(TRIPLET_GENERATORS.keys()), help='Which triplet generator to run with')

parser.add_argument('--base-model-name', type=str, default='', help='Base name for the models')

parser.add_argument('-m', '--model', type=str, action='append', choices=MODELS,
                    help='Which models to run')

parser.add_argument('--saycam', type=str, default=None, help='Which SAYcam model to use')
parser.add_argument('--imagenet-pretrained', action='store_true', help='Use imagenet pretrained models')
parser.add_argument('--untrained', action='store_true', help='Use untrained models')

parser.add_argument('-o', '--output-file', type=str, help='Output file to write to')

parser.add_argument('--rotate-angle', type=int, default=None, help='Angle to rotate the stimuli by')

parser.add_argument('--tqdm', action='store_true', help='Use tqdm progress bar')

parser.add_argument('--device', default=None, help='Which device to use. Defaults to cuda:0 if available and cpu if not.')

parser.add_argument('--profile', action='store_true', help='Profile')
parser.add_argument('--profile-output', default='/home/gd1279/scratch/profile/embeddings_profile')

MULTIPLE_OPTION_FIELD_DEFAULTS = {
    'relation': RELATIONS,
    'two_reference_objects': [0, 1],
    'adjacent_reference_objects': [0, 1],
    'n_target_types': [1, 2],
    'transpose_stimuli': [0, 1],
    'n_habituation_stimuli': [1, 4,],
    'rotate_angle': [None, 30, 45, 60],
}
MULTIPLE_OPTION_REWRITE_FIELDS = list(MULTIPLE_OPTION_FIELD_DEFAULTS.keys())

SINGLE_OPTION_FIELDS_TO_DF = ['seed', 'n_examples', 'extra_diagonal_margin']


def create_triplet_generators(args, name_func_kwargs=None):
    if name_func_kwargs is None:
        name_func_kwargs = {}
    
    triplet_generator_class = TRIPLET_GENERATORS[args.triplet_generator]

    triplet_generators = []

    for stimulus_generator_name in args.stimulus_generators:
        stimulus_generator_builder = STIMULUS_GENERATORS[stimulus_generator_name]        
        stimulus_generator = stimulus_generator_builder(**args.stimulus_generator_kwargs)

        triplet_generator = triplet_generator_class(stimulus_generator,args.distance_endpoints,
            relation=args.relation, two_reference_objects=args.two_reference_objects, 
            adjacent_reference_objects=args.adjacent_reference_objects,
            n_target_types=args.n_target_types, transpose=args.transpose_stimuli,
            n_habituation_stimuli=args.n_habituation_stimuli, 
            multiple_habituation_radius=args.multiple_habituation_radius,
            extra_diagonal_margin=args.extra_diagonal_margin)
        
        triplet_generators.append(triplet_generator)

    return triplet_generators
    

def handle_multiple_option_defaults(args):
    var_args = vars(args)
    for key in MULTIPLE_OPTION_FIELD_DEFAULTS:
        if key not in var_args or var_args[key] is None or (hasattr(var_args[key], '__len__') and len(var_args[key]) == 0):
            var_args[key] = MULTIPLE_OPTION_FIELD_DEFAULTS[key]

        elif not hasattr(var_args[key], '__len__'):
            var_args[key] = [var_args[key]]

    return args


def handle_single_args_setting(args):
    var_args = vars(args)
    
    model_kwarg_dicts = []
    for model_name in args.model:
        if args.saycam:
            model_kwarg_dicts.append(dict(name=model_name, device=args.device, pretrained=False, saycam=args.saycam))
        
        if args.imagenet_pretrained:
            model_kwarg_dicts.append(dict(name=model_name, device=args.device, pretrained=True))

        if args.untrained:
            model_kwarg_dicts.append(dict(name=model_name, device=args.device, pretrained=False))

    model_names = [ f'{d["name"]}-{"saycam({s})".format(s=d["syacam"]) if "saycam" in d else (d["pretrained"] and "imagenet" or "random")}'
               for d in model_kwarg_dicts]

    var_args['stimulus_generator_kwargs'] = {s.split('=')[0]: s.split('=')[1] for s in args.stimulus_generator_kwargs}

    all_model_results = []

    rep_iter = trange(args.replications) if args.tqdm else range(args.replications)
    for r in rep_iter:
        var_args['seed'] = args.seed + 1
        torch.manual_seed(args.seed)
        var_args['stimulus_generator_kwargs']['rng'] = np.random.default_rng(args.seed)
        var_args['stimulus_generator_kwargs']['rotate_angle'] = args.rotate_angle

        if args.distance_endpoints == DEFAULT_DISTANCE_ENDPOINTS:
            var_args['distance_endpoints'] = DISTANCE_ENDPOINTS_DICT[bool(args.two_reference_objects), bool(args.adjacent_reference_objects)]

        triplet_generators = create_triplet_generators(args)

        all_model_results.append(run_multiple_models_multiple_generators(
            model_names, model_kwarg_dicts, args.stimulus_generators, triplet_generators, args.n_examples))

    result_df = multiple_results_to_df(all_model_results, N=args.n_examples)

    for key in MULTIPLE_OPTION_REWRITE_FIELDS + SINGLE_OPTION_FIELDS_TO_DF:
        result_df[key] = var_args[key]

    return result_df


if __name__ == '__main__':
    main_args = parser.parse_args()
    main_args = handle_multiple_option_defaults(main_args)
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

    multiple_option_field_values = [main_var_args[key] for key in MULTIPLE_OPTION_REWRITE_FIELDS]

    dataframes = []

    value_iter = itertools.product(*multiple_option_field_values)
    if main_args.tqdm:
        total = np.prod([len(v) for v in multiple_option_field_values])
        value_iter = tqdm(value_iter, desc='Setting', total=total)

    for i, value_combination in enumerate(value_iter):
        args_copy = copy.deepcopy(main_args)
        var_args_copy = vars(args_copy)
        var_args_copy.update({key: value for key, value in zip(MULTIPLE_OPTION_REWRITE_FIELDS,
                                                               value_combination)})

        # TODO: any checks for arg combinations we shouldn't run?
        if args_copy.relation == ABOVE_BELOW_RELATION and not args_copy.two_reference_objects:
            print(f'Skpping because between relation and two_reference_objects={args_copy.two_reference_objects} is not set')
            continue

        if args_copy.adjacent_reference_objects and (args_copy.relation != ABOVE_BELOW_RELATION or not args_copy.two_reference_objects):
            print(f'Skpping because adjacent_reference_objects={args_copy.adjacent_reference_objects} is set and relation={args_copy.relation} is not above/below or two_reference_objects={args_copy.two_reference_objects} is not set')
            continue

        if main_args.profile:
            print('Profiling...')
            cProfile.run('handle_single_args_setting(args_copy)', f'{main_args.profile_output}_{i}')
        else:
            dataframes.append(handle_single_args_setting(args_copy))

    out_df = pd.concat(dataframes)
    out_df.reset_index(drop=True, inplace=True)
    out_df.to_csv(main_args.output_file)
