import argparse
import os
import random
import sys

sys.path.append(os.path.abspath('..'))

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from simple_relational_reasoning import datagen
from simple_relational_reasoning.datagen import object_gen
from simple_relational_reasoning import models


parser = argparse.ArgumentParser()

# Running-related arguments

DEFAULT_SEED = 33
parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed to run with')

DEFAULT_MAX_EPOCHS = 10000
parser.add_argument('--max-epochs', type=int, default=DEFAULT_MAX_EPOCHS, help='After how many epochs should we stop')

DEFAULT_PATIENCE_EPOCHS = 50
parser.add_argument('--patience-epochs', type=int, default=DEFAULT_PATIENCE_EPOCHS,
                    help='How many patience epochs (stop after this many epochs with no improvement)')

DEFAULT_BATCH_SIZE = 2 ** 10
parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                    help='Batch size to run with')

DEFAULT_DATASET_SIZE = 2 ** 14
parser.add_argument('--dataset-size', type=int, default=DEFAULT_DATASET_SIZE,
                    help='Dataset size to generate')

parser.add_argument('--validation-size', type=int, default=None,
                    help='Validation size to generate (if different from regular dataset)')

DEFAULT_LEARNING_RATE = 1e-3
parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE,
                    help='Learning rate to run with')

# Relation-related arguments

parser.add_argument('--num-objects', type=int, required=True, help='How many objects in each collection/scene')

RELATION_NAMES_TO_CLASSES = {
    'adjacent': datagen.MultipleDAdjacentRelation,
    'above': datagen.ColorAboveColorRelation,
    'count': datagen.ObjectCountRelation
}
parser.add_argument('--relation', type=str, default=None, choices=list(RELATION_NAMES_TO_CLASSES.keys()),
                    help='Which relation to use')

MODEL_CONFIGURATIONS = {
    'default': {
        models.CombinedObjectMLPModel: dict(embedding_size=8, prediction_sizes=[32, 32]),
        models.RelationNetModel: dict(embedding_size=8, object_pair_layer_sizes=[32], combined_object_layer_sizes=[32]),
        models.TransformerModel: dict(embedding_size=8, transformer_mlp_sizes=[8], mlp_sizes=[32]),
        models.CNNModel: dict(conv_sizes=[16, 16], conv_output_size=256)
    }
}


parser.add_argument('--model-configuration', type=str, required=True, choices=list(MODEL_CONFIGURATIONS.keys()),
                    help='Which model configuration to use')

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


CLASS_NAME_SPLIT_WORDS = ('net', 'object', 'mlp')


def prettify_class_name(cls):
    name = cls.__name__.lower().replace('model', '')
    for word in CLASS_NAME_SPLIT_WORDS:
        name = name.replace(word, f'-{word}')

    return name


def run_single_relation(args):
    # TODO: create dataset
    field_configs = FIELD_CONFIGURATIONS[args.field_configuration]
    relation_class = RELATION_NAMES_TO_CLASSES[args.relation]
    object_generator = object_gen.SmartBalancedBatchObjectGenerator(args.num_objects, field_configs, relation_class,
                                                                    max_recursion_depth=args.max_recursion_depth)

    train_dataset = datagen.ObjectGeneratorDataset(object_generator, args.dataset_size)
    if args.validation_size is None:
        args.validation_size = args.dataset_size

    validation_dataset = datagen.ObjectGeneratorDataset(object_generator, args.validation_size)

    # TODO: for each model configuration
    model_configurations = MODEL_CONFIGURATIONS[args.model_configuration]

    for model_class, model_kwargs in model_configurations.items():
        # TODO: add in learning rate, batch size, dataset size to the per-model kwargs
        model_kwargs['lr'] = args.learning_rate
        model_kwargs['batch_size'] = args.batch_size
        model_kwargs['train_epoch_size'] = args.dataset_size
        model_kwargs['validation_epoch_size'] = args.dataset_size
        model_kwargs['regenerate_every_epoch'] = False
        model_kwargs['train_dataset'] = train_dataset
        model_kwargs['validation_dataset'] = validation_dataset

        # TODO: create wandb project name
        args.wandb_project = f'{args.relation}-relation-{args.model_configuration}-models-{args.num_objects}-objects-{args.dataset_size}-dataset'

        # TODO: create wandb run with name appropriate for model and random seed
        args.wandb_run_name = f'{prettify_class_name(model_class)}-{args.seed}'

        # TODO: create model
        model = model_class(object_generator, **model_kwargs)

        args.total_params = sum(p.numel() for p in model.parameters())
        print(f'For {model_class.__name__} there are {args.total_params} total parameters')

        args.save_folder = os.path.join(SCRATCH_FOLDER, 'simple-relational-reasoning-checkpoints', args.wandb_project,
                                        args.wandb_run_name)

        args.use_gpu = int(torch.cuda.is_available())

        logger = WandbLogger(args.wandb_run_name, args.wandb_dir, project=args.wandb_project,
                             entity=args.wandb_entity, log_model=True)
        logger.log_hyperparams(vars(args))

        checkpoint_callback = ModelCheckpoint(filepath=os.path.join(args.save_folder, f'{args.wandb_run_name}_epoch-{{epoch:d}}_val-loss-{{val_loss:.3f}}'),
                                              save_top_k=2, verbose=True, monitor='val_loss', mode='min')
        early_stopping_callback = EarlyStopping('val_loss', patience=args.patience_epochs, verbose=True)

        # TODO: run with wandb logger
        trainer = Trainer(logger=logger, gpus=args.use_gpu, max_epochs=args.max_epochs,
                          checkpoint_callback=checkpoint_callback, early_stop_callback=early_stopping_callback)

        trainer.fit(model)

        del trainer
        del model


def main():
    args = parser.parse_args()

    # Handle slurm array ids
    array_id = os.getenv('SLURM_ARRAY_TASK_ID')
    if array_id is not None:
        args.seed = args.seed + int(array_id)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.relation is not None and len(args.relation) > 0:
        print(f'Running single relation: {args.relation}')
        run_single_relation(args)

    else:
        for relation in RELATION_NAMES_TO_CLASSES:
            args.relation = relation
            print(f'Running all {len(RELATION_NAMES_TO_CLASSES)} relations, current relation: {args.relation}')
            run_single_relation(args)


if __name__ == '__main__':
    main()


