import argparse
import os
import sys

sys.path.append(os.path.abspath('..'))

from simple_relational_reasoning import datagen
from simple_relational_reasoning.datagen import object_gen
from simple_relational_reasoning import models


parser = argparse.ArgumentParser()

# Running-related arguments

DEFAULT_SEED = 33
parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed to run with')

DEFAULT_MAX_EPOCHS = 10000
parser.add_argument('--max-epochs', type=int, default=DEFAULT_MAX_EPOCHS, help='After how many epochs should we stop')

DEFAULT_BATCH_SIZE = 2 ** 10
parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                    help='Batch size to run with')

DEFAULT_DATASET_SIZE = 2 ** 14
parser.add_argument('--dataset-size', type=int, action='append',
                    help='Dataset size to generate')

parser.add_argument('--validation-size', type=int, default=None,
                    help='Validation size to generate (if different from regular dataset)')

parser.add_argument('--test-size', type=int, default=None,
                    help='Test size to generate (if different from regular dataset)')

DEFAULT_LEARNING_RATE = 1e-3
parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE,
                    help='Learning rate to run with')

DEFAULT_PATIENCE_EPOCHS = 50
parser.add_argument('--patience-epochs', type=int, default=DEFAULT_PATIENCE_EPOCHS,
                    help='How many patience epochs (stop after this many epochs with no improvement)')

DEFAULT_MIN_DELTA = 1e-5
parser.add_argument('--early-stopping-min-delta', type=float, default=DEFAULT_MIN_DELTA,
                    help='What minimal improvement in the metric to consider as an actualy improvement')

# Relation-related arguments
DEFAULT_NUM_OBJECTS = 5
parser.add_argument('--num-objects', type=int, action='append', help='How many objects in each collection/scene')

RELATION_NAMES_TO_CLASSES = {
    'adjacent': datagen.MultipleDAdjacentRelation,
    'above': datagen.ColorAboveColorRelation,
    'count': datagen.ObjectCountRelation,
    'between': datagen.BetweenRelation,
}
parser.add_argument('--relation', type=str, action='append', choices=list(RELATION_NAMES_TO_CLASSES.keys()),
                    help='Which relation(s) to run (default: all)')

DEFAULT_MODELS_CONFIG_KEY = 'default'
LARGER_MODELS_CONFIG_KEY = 'larger'
MODEL_CONFIGURATIONS = {
    DEFAULT_MODELS_CONFIG_KEY: {
        models.CombinedObjectMLPModel: dict(embedding_size=8, prediction_sizes=[32, 32]),
        models.RelationNetModel: dict(embedding_size=8, object_pair_layer_sizes=[32], combined_object_layer_sizes=[32]),
        models.TransformerModel: dict(embedding_size=8, transformer_mlp_sizes=[8], mlp_sizes=[32]),
        models.CNNModel: dict(conv_sizes=[16, 16], conv_output_size=256),
        # models.FixedCNNModel: dict(conv_sizes=[6, 6], conv_output_size=96, mlp_sizes=[16, 16],),
    },
    LARGER_MODELS_CONFIG_KEY: {
        models.CombinedObjectMLPModel: dict(embedding_size=16, prediction_sizes=[64, 32, 16]),
        models.RelationNetModel: dict(embedding_size=16, object_pair_layer_sizes=[64, 32],
                                      combined_object_layer_sizes=[64, 32]),
        models.TransformerModel: dict(embedding_size=16, num_transformer_layers=2, num_heads=2,
                                      transformer_mlp_sizes=[32, 16], mlp_sizes=[64, 32]),
        models.CNNModel: dict(conv_sizes=[8, 12, 16], conv_output_size=64, mlp_sizes=[32, 32, 32],),
        # models.FixedCNNModel: dict(conv_sizes=[6, 6], conv_output_size=96, mlp_sizes=[16, 16],),
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


FIELD_CONFIGURATIONS = {
    'default': (
        object_gen.FieldConfig('x', 'int_position', dict(max_coord=16)),
        object_gen.FieldConfig('y', 'int_position', dict(max_coord=16)),
        object_gen.FieldConfig('color', 'one_hot', dict(n_types=2)),
        object_gen.FieldConfig('shape', 'one_hot', dict(n_types=2))
    )
}
parser.add_argument('--field-configuration', type=str, required=True, choices=list(FIELD_CONFIGURATIONS.keys()),
                    help='Which field configuration to use')


DEFAULT_OBJECT_GENERATOR_MAX_RECURSION_DEPTH = 1000
parser.add_argument('--max-recursion-depth', type=int, default=DEFAULT_OBJECT_GENERATOR_MAX_RECURSION_DEPTH,
                    help='Maximal recursion depth when trying to generate objects')


# Wandb-related arguments
SCRATCH_FOLDER = r'/misc/vlgscratch4/LakeGroup/guy/'

DEFUALT_WANDB_ENTITY = 'simple-relational-reasoning'
parser.add_argument('--wandb-entity', default=DEFUALT_WANDB_ENTITY)

DEFAULT_WANDB_DIR = SCRATCH_FOLDER  # wandb creates its own folder inside
parser.add_argument('--wandb-dir', default=DEFAULT_WANDB_DIR)
parser.add_argument('--wandb-omit-watch', action='store_true')


# Handling fields with multiple potential options
MULTIPLE_OPTION_FIELD_DEFAULTS = {
    'model_configuration': [DEFAULT_MODELS_CONFIG_KEY],
    'dataset_size': [DEFAULT_DATASET_SIZE],
    'num_objects': [DEFAULT_NUM_OBJECTS],
    'relation': list(RELATION_NAMES_TO_CLASSES.keys()),
    'model': MODEL_NAMES
}
MULTIPLE_OPTION_REWRITE_FIELDS = list(MULTIPLE_OPTION_FIELD_DEFAULTS.keys())
MULTIPLE_OPTION_REWRITE_FIELDS.remove('model')  # intentionally no rewrite the model field downstream


def handle_multiple_option_defaults(args):
    var_args = vars(args)
    for key in MULTIPLE_OPTION_FIELD_DEFAULTS:
        if key not in var_args or var_args[key] is None or len(var_args[key]) == 0:
            var_args[key] = MULTIPLE_OPTION_FIELD_DEFAULTS[key]

    return args

