import copy
import itertools
import os
import random
import sys
import wandb

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from simple_relational_reasoning import datagen
from simple_relational_reasoning.datagen.quinn_objects import *

from quinn_defaults import *


def run_single_setting_all_models(args):
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

    # TODO: for each model configuration
    model_configurations = MODEL_CONFIGURATIONS[args.model_configuration]

    for model_class, model_kwargs in model_configurations.items():
        model_class_name = prettify_class_name(model_class)
        if args.model is not None and len(args.model) > 0 and model_class_name not in args.model:
            print(f'Skipping model {model_class_name} because it is not in {args.model}')
            continue

        args.model_name = model_class_name
        args.spatial_dataset = 'cnn' in args.model_name
        object_generator, dataset = create_dataset(args)

        # TODO: add in learning rate, batch size, dataset size to the per-model kwargs
        model_kwargs['dataset'] = dataset
        model_kwargs['lr'] = args.learning_rate
        model_kwargs['batch_size'] = args.batch_size

        # TODO: create wandb project name
        args.wandb_project = f'{args.paradigm}-{args.relation}-{args.model_configuration}-models{"-" if args.wandb_project_suffix else ""}{args.wandb_project_suffix}'

        # TODO: create wandb run with name appropriate for model and random seed
        args.wandb_run_name = f'{model_class_name}-{args.seed}'

        # TODO: create model
        model = model_class(object_generator, **model_kwargs)
        args.use_gpu = int(torch.cuda.is_available())
        args.total_params = sum(p.numel() for p in model.parameters())
        print(f'For {model_class.__name__} there are {args.total_params} total parameters')

        logger = WandbLogger(args.wandb_run_name, args.wandb_dir, project=args.wandb_project,
                             entity=args.wandb_entity, log_model=True)
        logger.log_hyperparams(vars(args))

        # TODO: is this supposed to work without this hack?
        # logger.experiment creates and returns the wandb run
        wandb.save(os.path.join(logger.experiment.dir, '*.ckpt'))

        checkpoint_callback = ModelCheckpoint(filepath=os.path.join(wandb.run.dir, f'{args.wandb_run_name}-{{epoch:d}}-{{val_loss:.3f}}'),
                                              save_top_k=1, verbose=True, monitor='val_loss', mode='min')
        early_stopping_callback = EarlyStopping('val_loss', patience=args.patience_epochs, verbose=True,
                                                min_delta=args.early_stopping_min_delta)

        # TODO: run with wandb logger
        trainer = Trainer(logger=logger, gpus=args.use_gpu, max_epochs=args.max_epochs,
                          checkpoint_callback=checkpoint_callback, early_stop_callback=early_stopping_callback)

        trainer.fit(model)

        logger.save()
        logger.close()

        del trainer
        del model
        # Should be unnecessary now that I'm not keeping a handle to the run object
        # if run is not None:
        #     run.close_files()
        #     del run


def create_dataset(args):
    # TODO: create object generator
    if args.use_object_size:
        object_generator_class = ObjectGeneratorWithSize
    else:
        object_generator_class = ObjectGeneratorWithoutSize
    object_generator = object_generator_class(args.seed, args.reference_object_length,
                                              args.target_object_length, args.n_reference_object_types,
                                              args.n_train_target_object_types, args.n_test_target_object_types)
    # TODO: create dataset from paradigm and relation
    if args.paradigm == 'inductive_bias':
        if args.relation == 'above_below':
            dataset_class = AboveBelowReferenceInductiveBias

        else:  # args.relation == 'between
            dataset_class = BetweenReferenceInductiveBias

        dataset = dataset_class(
            object_generator, args.x_max, args.y_max, args.seed, args.target_object_grid_size,
            args.add_neither_train, args.above_or_between_left,
            args.n_train_target_object_locations, args.prop_train_reference_object_locations,
            args.reference_object_x_margin, args.reference_object_bottom_y_margin,
            args.reference_object_top_y_margin, args.add_neither_test, args.spatial_dataset
        )

    else:  # args.paradigm == 'one_or_two_references':
        dataset = OneOrTwoReferenceObjects(
            object_generator, args.x_max, args.y_max, args.seed, args.relation == 'between',
            args.two_reference_object, args.add_neither_train, args.above_or_between_left,
            args.n_train_target_object_locations, args.prop_train_reference_object_locations,
            args.reference_object_x_margin, args.reference_object_bottom_y_margin,
            args.reference_object_top_y_margin, args.add_neither_test, args.spatial_dataset
        )
    return object_generator, dataset


def main():
    main_args = parser.parse_args()

    # Handle slurm array ids
    array_id = os.getenv('SLURM_ARRAY_TASK_ID')
    if array_id is not None:
        main_args.seed = main_args.seed + int(array_id)

    random.seed(main_args.seed)
    torch.manual_seed(main_args.seed)
    torch.cuda.manual_seed_all(main_args.seed)

    main_args = handle_multiple_option_defaults(main_args)
    main_var_args = vars(main_args)
    print(' ' * 26 + 'Global Options')
    for k, v in main_var_args.items():
        print(' ' * 26 + k + ': ' + str(v))

    multiple_option_field_values = [main_var_args[key] for key in MULTIPLE_OPTION_REWRITE_FIELDS]

    for value_combination in itertools.product(*multiple_option_field_values):
        args_copy = copy.deepcopy(main_args)
        var_args_copy = vars(args_copy)
        var_args_copy.update({key: value for key, value in zip(MULTIPLE_OPTION_REWRITE_FIELDS,
                                                               value_combination)})

        # hack to make sure I don't run a particular case that doesn't make sense
        if not args_copy.add_neither_train and args_copy.add_neither_test:
            continue

        run_single_setting_all_models(args_copy)


if __name__ == '__main__':
    main()

