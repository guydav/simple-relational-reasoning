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
from simple_relational_reasoning.datagen import object_gen
from simple_relational_reasoning import models

from defaults import *


def run_single_setting_all_models(args):
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

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
        model_class_name = prettify_class_name(model_class)
        if args.model is not None and len(args.model) > 0 and model_class_name not in args.model:
            print(f'Skipping model {model_class_name} because it is not in {args.model}')
            continue

        args.model_name = model_class_name

        # TODO: add in learning rate, batch size, dataset size to the per-model kwargs
        model_kwargs['lr'] = args.learning_rate
        model_kwargs['batch_size'] = args.batch_size
        model_kwargs['train_epoch_size'] = args.dataset_size
        model_kwargs['validation_epoch_size'] = args.validation_size
        model_kwargs['regenerate_every_epoch'] = False
        model_kwargs['train_dataset'] = train_dataset
        model_kwargs['validation_dataset'] = validation_dataset

        # TODO: create wandb project name
        args.wandb_project = f'{args.relation}-relation-{args.model_configuration}-models-{args.num_objects}-objects-{args.dataset_size}-dataset'

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

        run_single_setting_all_models(args_copy)


if __name__ == '__main__':
    main()

