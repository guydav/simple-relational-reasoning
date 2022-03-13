import argparse
import os
import sys

sys.path.append(os.path.abspath('..'))

from simple_relational_reasoning import models


parser = argparse.ArgumentParser()

# Running-related arguments

DEFAULT_SEED = 33
parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed to run with')

DEFAULT_MAX_EPOCHS = 10000
parser.add_argument('--max-epochs', type=int, default=DEFAULT_MAX_EPOCHS, help='After how many epochs should we stop')

DEFAULT_BATCH_SIZE = 2 ** 8
parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                    help='Batch size to run with')

DEFAULT_LEARNING_RATE = 1e-3
parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE,
                    help='Learning rate to run with')

DEFAULT_PATIENCE_EPOCHS = 50
parser.add_argument('--patience-epochs', type=int, default=DEFAULT_PATIENCE_EPOCHS,
                    help='How many patience epochs (stop after this many epochs with no improvement)')

DEFAULT_MIN_DELTA = 1e-5
parser.add_argument('--early-stopping-min-delta', type=float, default=DEFAULT_MIN_DELTA,
                    help='What minimal improvement in the metric to consider as an actualy improvement')

# Quinn object generator arguments

# parser.add_argument('--use-object-size', type=int, default=None, help='Whether or not to use object sizes')

parser.add_argument('--use-start-end', type=int, default=None, help='Whether or not to use start and end object representations') 

parser.add_argument('--adjacent-reference-objects', type=int, default=None, help='Whether or not to use adjacent reference objects (in the above/below, two-reference object case')

DEFAULT_REFERENCE_OBJECT_LENGTH = 9
parser.add_argument('--reference-object-length', type=int, default=DEFAULT_REFERENCE_OBJECT_LENGTH,
                    help='Reference object (horizontal) length')

DEFAULT_TARGET_OBJECT_LENGTH = 1
parser.add_argument('--target-object-length', type=int, default=DEFAULT_TARGET_OBJECT_LENGTH,
                    help='Target object (horizontal) length')

DEFAULT_N_REFERENCE_OBJECT_TYPES = 1
parser.add_argument('--n-reference-object-types', type=int, default=DEFAULT_N_REFERENCE_OBJECT_TYPES,
                    help='Number of reference object types to use')

DEFAULT_N_TRAIN_TARGET_OBJECT_TYPES = 1
parser.add_argument('--n-train-target-object-types', type=int, default=DEFAULT_N_TRAIN_TARGET_OBJECT_TYPES,
                    help='Number of target object types to use in training (and in test if n_train_target_object_types = 0)')

DEFAULT_N_TEST_TARGET_OBJECT_TYPES = 0
parser.add_argument('--n-test-target-object-types', type=int, default=DEFAULT_N_TEST_TARGET_OBJECT_TYPES,
                    help='Number of target object types to use in test')

# Quinn dataset arguments

DEFAULT_X_MAX = 18
parser.add_argument('--x_max', type=int, default=DEFAULT_X_MAX, help='Canvas X size')

DEFAULT_Y_MAX = 18
parser.add_argument('--y_max', type=int, default=DEFAULT_Y_MAX, help='Canvas Y size')

DEFAULT_PROP_TRAIN_REFERENCE_OBJECT_LOCATIONS = 0.9
parser.add_argument('--prop-train-reference-object-locations', type=float,
                    default=DEFAULT_PROP_TRAIN_REFERENCE_OBJECT_LOCATIONS,
                    help='Proportion of reference object locations to assign to the training set')

DEFAULT_REFERENCE_OBJECT_X_MARGIN = 0
parser.add_argument('--reference-object-x-margin', type=int, default=DEFAULT_REFERENCE_OBJECT_X_MARGIN,
                    help='Horizontal margin to allow between the reference object and the edge of the canvas')

DEFAULT_REFERENCE_OBJECT_Y_MARGIN_BOTTOM = None
parser.add_argument('--reference-object-y-margin-bottom', type=int, default=DEFAULT_REFERENCE_OBJECT_Y_MARGIN_BOTTOM,
                    help='Bottom vertical margin to allow between the reference object and the edge of the canvas')

DEFAULT_REFERENCE_OBJECT_Y_MARGIN_TOP = None
parser.add_argument('--reference-object-y-margin-top', type=int, default=DEFAULT_REFERENCE_OBJECT_Y_MARGIN_TOP,
                    help='Top vertical margin to allow between the reference object and the edge of the canvas')

DEFAULT_PROP_TRAIN_ASSIGNED_TO_VALIDATION = 0.1
parser.add_argument('--prop-train-to-validation', type=float,
                    default=DEFAULT_PROP_TRAIN_ASSIGNED_TO_VALIDATION,
                    help='Proportion of training set to assign as a validation set')

DEFAULT_EARLY_STOPPING_MONITOR_KEY = 'val'
parser.add_argument('--early-stopping-monitor-key', type=str, default=DEFAULT_EARLY_STOPPING_MONITOR_KEY,
                    help='Which key to monitor for validation/test stopping')

SUBSAMPLE_FULL_DATASET = -1
parser.add_argument('--subsample-train-size', type=int, default=None,
                    help='How much to subsample the training set to (default is None to keep the full set)')



ABOVE_BELOW_RELATION = 'above_below'
BETWEEN_RELATION = 'between'
DIAGONAL_RELATION = 'diagonal'

RELATIONS = (
    ABOVE_BELOW_RELATION,
    BETWEEN_RELATION,
    DIAGONAL_RELATION
)
parser.add_argument('--relation', type=str, action='append', choices=RELATIONS,
                    help='Which relation(s) to run (default: all)')

DEFAULT_TWO_REFERENCE_OBJECTS = None
parser.add_argument('--two-reference-objects', type=int, default=DEFAULT_TWO_REFERENCE_OBJECTS,
                    help='Whether or not to use two reference objects. Defaults to True for the between relation and False for above/below.')

DEFAULT_PROP_TRAIN_TARGET_OBJECT_LOCATIONS = 0.8
parser.add_argument('--prop-train-target-object-locations', type=float,
                    default=DEFAULT_PROP_TRAIN_TARGET_OBJECT_LOCATIONS,
                    help='Proportion of target object locations to assign to the training set')

DEFAULT_TARGET_OBJECT_GRID_HEIGHT = 8
parser.add_argument('--target-object-grid-height', type=int, default=DEFAULT_TARGET_OBJECT_GRID_HEIGHT,
                    help='How big of a grid to assign target objects in')


# Model setup arguments

DEFAULT_MODELS_CONFIG_KEY = 'default'
LARGER_MODELS_CONFIG_KEY = 'larger'
MODEL_CONFIGURATIONS = {
    DEFAULT_MODELS_CONFIG_KEY: {
        models.CombinedObjectMLPModel: dict(embedding_size=8, prediction_sizes=[32, 32, 16]),
        models.RelationNetModel: dict(embedding_size=8, object_pair_layer_sizes=[32], combined_object_layer_sizes=[32]),
        models.TransformerModel: dict(embedding_size=8, transformer_mlp_sizes=[8, 8], mlp_sizes=[32, 32]),
        # models.CNNModel: dict(conv_sizes=[8, 16], mlp_sizes=[16, 8],),
        models.SimplifiedCNNModel: dict(conv_sizes=[8, 16], mlp_sizes=[16, 8],),
        models.PrediNetModel: dict(key_size=4, num_heads=4, num_relations=4, output_hidden_size=16),
        models.PrediNetWithEmbeddingModel: dict(embedding_size=8, key_size=4, num_heads=2, num_relations=4, output_hidden_size=16),
    },
    LARGER_MODELS_CONFIG_KEY: {
        models.CombinedObjectMLPModel: dict(embedding_size=16, prediction_sizes=[64, 64, 32, 16]),
        models.RelationNetModel: dict(embedding_size=16, object_pair_layer_sizes=[64, 32],
                                      combined_object_layer_sizes=[64, 32]),
        models.TransformerModel: dict(embedding_size=16, num_transformer_layers=3, num_heads=2,
                                      transformer_mlp_sizes=[16, 16], mlp_sizes=[64, 32]),
        # models.CNNModel: dict(conv_sizes=[8, 16, 32], mlp_sizes=[32, 32],),
        models.SimplifiedCNNModel: dict(conv_sizes=[8, 16, 32], mlp_sizes=[32, 32],),
        models.PrediNetModel: dict(key_size=8, num_heads=8, num_relations=8, output_hidden_size=32),
        models.PrediNetWithEmbeddingModel: dict(embedding_size=16, key_size=6, num_heads=4, num_relations=6, output_hidden_size=16),
    }
}
parser.add_argument('--model-configuration', type=str, action='append', choices=list(MODEL_CONFIGURATIONS.keys()),
                    help='Which model configuration to use')


CLASS_NAME_SPLIT_WORDS = ('net', 'object', 'mlp', 'cnn')


def prettify_class_name(cls):
    name = cls.__name__.lower().replace('model', '')
    for word in CLASS_NAME_SPLIT_WORDS:
        name = name.replace(word, f'-{word}')

    if name.startswith('-'):
        return name[1:]

    return name


MODEL_NAMES = [prettify_class_name(model_class) for model_class in MODEL_CONFIGURATIONS['default'].keys()]
parser.add_argument('--model', type=str, action='append', choices=MODEL_NAMES,
                    help='Which model(s) to use (default: all)')


# Wandb-related arguments
SCRATCH_FOLDER = r'/misc/vlgscratch4/LakeGroup/guy/'

DEFUALT_WANDB_ENTITY = 'quinn-relations-v1'
parser.add_argument('--wandb-entity', default=DEFUALT_WANDB_ENTITY)

DEFAULT_WANDB_DIR = SCRATCH_FOLDER  # wandb creates its own folder inside
parser.add_argument('--wandb-dir', default=DEFAULT_WANDB_DIR)
parser.add_argument('--wandb-omit-watch', action='store_true')

parser.add_argument('--wandb-name-suffix', type=str, default='')
parser.add_argument('--wandb-project-suffix', type=str, default='')
parser.add_argument('--wandb-project', type=str, default=None)


# Handling fields with multiple potential options
MULTIPLE_OPTION_FIELD_DEFAULTS = {
    'subsample_train_size': [8, 32, 128, 512, 1024, 2048, SUBSAMPLE_FULL_DATASET],
    'model_configuration': [DEFAULT_MODELS_CONFIG_KEY, LARGER_MODELS_CONFIG_KEY],
    'model': MODEL_NAMES,
    'relation': RELATIONS,
    'use_start_end': [0, 1],
    'two_reference_objects': [0, 1],
    'adjacent_reference_objects': [0, 1],
}

MULTIPLE_OPTION_REWRITE_FIELDS = list(MULTIPLE_OPTION_FIELD_DEFAULTS.keys())
MULTIPLE_OPTION_REWRITE_FIELDS.remove('model')  # intentionally no rewrite of the model field downstream


def handle_multiple_option_defaults(args):
    var_args = vars(args)
    for key in MULTIPLE_OPTION_FIELD_DEFAULTS:
        if key not in var_args or var_args[key] is None or (hasattr(var_args[key], '__len__') and len(var_args[key]) == 0):
            var_args[key] = MULTIPLE_OPTION_FIELD_DEFAULTS[key]

        elif not hasattr(var_args[key], '__len__'):
            var_args[key] = [var_args[key]]

    return args

