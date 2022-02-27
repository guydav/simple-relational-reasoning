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

    if args.x_max == DEFAULT_X_MAX:
        args.x_max = DEFAULT_CANVAS_SIZES[args.relation]['x_max']

    if args.y_max == DEFAULT_Y_MAX:
        args.y_max = DEFAULT_CANVAS_SIZES[args.relation]['y_max']

    # for each model
    model_configurations = MODEL_CONFIGURATIONS[args.model_configuration]

    for model_class, model_kwargs in model_configurations.items():
        model_class_name = prettify_class_name(model_class)
        if args.model is not None and len(args.model) > 0 and model_class_name not in args.model:
            print(f'Skipping model {model_class_name} because it is not in {args.model}')
            continue

        if model_class_name == 'simplified-cnn' and args.use_object_size:
            print(f'Skiping running {model_class_name} with args.use_object_size={args.use_object_size} as they are incompatible')
            continue

        args.model_name = model_class_name
        if 'simplified' in args.model_name.lower():
            args.spatial_dataset = 'simplified'
        else:
            args.spatial_dataset = 'cnn' in args.model_name.lower()

        _, dataset = create_dataset(args)

        args.size_train_dataset = len(dataset.get_training_dataset())
        args.size_val_dataset = len(dataset.get_validation_dataset().labels)
        var_args = vars(args)
        for test_name, test_dataset in dataset.get_test_datasets().items():
            var_args[f'size_{test_name}'] = len(test_dataset)

        # add in learning rate, batch size, dataset size to the per-model kwargs
        model_kwargs['lr'] = args.learning_rate
        model_kwargs['batch_size'] = args.batch_size

        # create wandb project name
        map_args_to_suffix(args)
        if args.wandb_project is None:
            args.wandb_project = f'one_or_two_references-{args.relation}-{args.model_configuration}-models{"-" if args.wandb_project_suffix else ""}{args.wandb_project_suffix}'

        # create wandb run with name appropriate for model and random seed
        args.wandb_run_name = f'{model_class_name}-{args.seed}{"-" if args.wandb_name_suffix else ""}{args.wandb_name_suffix}'

        # create model
        model = model_class(dataset, **model_kwargs)
        args.use_gpu = int(torch.cuda.is_available())
        args.total_params = sum(p.numel() for p in model.parameters())
        print(f'For {model_class.__name__} there are {args.total_params} total parameters')

        logger = WandbLogger(args.wandb_run_name, args.wandb_dir, project=args.wandb_project,
                             entity=args.wandb_entity, log_model=True)
        logger.log_hyperparams(vars(args))

        checkpoint_callback = ModelCheckpoint(dirpath=wandb.run.dir, filename=f'{args.wandb_run_name}-{{epoch:d}}-{{val_loss:.5f}}',
                                              save_top_k=1, verbose=True, monitor=args.early_stopping_monitor_key, mode='min')
        early_stopping_callback = EarlyStopping(args.early_stopping_monitor_key, patience=args.patience_epochs, verbose=True,
                                                min_delta=args.early_stopping_min_delta)

        # run with wandb logger
        trainer = Trainer(logger=logger, gpus=args.use_gpu, max_epochs=args.max_epochs,
                          callbacks=[checkpoint_callback, early_stopping_callback])

        trainer.fit(model)

        logger.save()
        logger.finalize('Done')

        wandb.finish()

        del trainer
        del model



def map_args_to_suffix(args):
    if args.wandb_project_suffix:
        return

    suffix_components = list()
    suffix_components.append(args.use_object_size and 'with-object-size' or 'without-object-size')
    suffix_components.append(args.add_neither_train and 'with-neither' or 'without-neither')

    args.wandb_project_suffix = '-'.join(suffix_components)


def create_dataset(args):
    if args.relation == DIAGONAL_RELATION:
        object_generator_class = DiagonalObjectGeneratorWithoutSize
    elif args.use_object_size:
        object_generator_class = ObjectGeneratorWithSize
    else:
        object_generator_class = ObjectGeneratorWithoutSize

    object_generator = object_generator_class(args.seed, args.reference_object_length,
                                              args.target_object_length, args.n_reference_object_types,
                                              args.n_train_target_object_types, args.n_test_target_object_types)
    
    if args.relation == DIAGONAL_RELATION:
        reference_object_y_margin = None
        if args.reference_object_y_margin_bottom is not None or args.reference_object_y_margin_top is not None:
            if args.reference_object_y_margin_bottom is not None and args.reference_object_y_margin_top is not None:
                reference_object_y_margin = max(args.reference_object_y_margin_bottom, args.reference_object_y_margin_top)

            else:
                reference_object_y_margin = args.reference_object_y_margin_bottom if args.reference_object_y_margin_bottom is not None else args.reference_object_y_margin_top

        dataset = DiagonalAboveBelowDatasetGenerator(
            object_generator, args.x_max, args.y_max, args.seed,
            reference_object_x_margin=args.reference_object_x_margin,
            reference_object_y_margin=reference_object_y_margin,
            prop_train_target_object_locations=args.prop_train_target_object_locations,
            prop_train_reference_object_locations=args.prop_train_reference_object_locations,
            spatial_dataset=args.spatial_dataset,
            prop_train_to_validation=args.prop_train_to_validation, 
            subsample_train_size=args.subsample_train_size
        )

    else:
        dataset = CombinedQuinnDatasetGenerator(
            object_generator, args.x_max, args.y_max, args.seed,
            between_relation=args.relation == BETWEEN_RELATION,
            two_reference_objects=args.two_reference_objects,
            prop_train_target_object_locations=args.prop_train_target_object_locations,
            prop_train_reference_object_locations=args.prop_train_reference_object_locations,
            target_object_grid_height=args.target_object_grid_height,
            reference_object_x_margin=args.reference_object_x_margin,
            reference_object_y_margin_bottom=args.reference_object_y_margin_bottom,
            reference_object_y_margin_top=args.reference_object_y_margin_top,
            spatial_dataset=args.spatial_dataset,
            prop_train_to_validation=args.prop_train_to_validation, 
            subsample_train_size=args.subsample_train_size
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
        args_copy.add_neither_test = args_copy.add_neither_train

        if not args_copy.early_stopping_monitor_key.endswith('_loss'):
            args_copy.early_stopping_monitor_key += '_loss'

        # remove a case that doesn't make sense
        if args_copy.relation == DIAGONAL_RELATION and (args_copy.two_reference_objects or args_copy.use_object_size):
            print(f'Skpping because diagonal relation and either two _reference_objects={args_copy.two_reference_objects} or use_object_size={args_copy.use_object_size} is set')
            continue

        run_single_setting_all_models(args_copy)


if __name__ == '__main__':
    main()

