import copy
import itertools
import os
import random
import sys
import wandb
import numpy as np

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger

from simple_relational_reasoning import datagen
from simple_relational_reasoning.datagen import object_gen
from simple_relational_reasoning import models

from defaults import *


parser.add_argument('--test-field', type=str, required=True, help='Which parameter to run a generalization test over')
parser.add_argument('--test-values', type=str, action='append', required=True,
                    help='Which values to test over (must have preexisting data with saved models and all that jazz)')

DEFAULT_LOG_PREFIX = 'num_objects_gen'
parser.add_argument('--test-log-prefix', type=str, default=DEFAULT_LOG_PREFIX,
                    help='Which prefix to log validation data with')

DEFAULT_CHECKPOINT_DONWLOAD_FOLDER = '/scratch/gd1279/simple-relational-reasoning-checkpoints'
parser.add_argument('--checkpoint-download-folder', type=str, default=DEFAULT_CHECKPOINT_DONWLOAD_FOLDER,
                    help='Where to download checkpoints to')


def run_generalization_test_single_setting(args):
    print(' ' * 26 + 'Options')
    var_args = vars(args)
    for k, v in var_args.items():
        print(' ' * 26 + k + ': ' + str(v))

    if args.test_field != 'num_objects':
        raise ValueError(f'Only implemented so far for num_objects, received test-field = {args.test_field}')

    # TODO: loop over assignment to test_value, model_test_value
    for trained_value, test_value in itertools.permutations(args.test_values, 2):
        args.num_objects = trained_value
        args.model_test_value = test_value

        field_configs = FIELD_CONFIGURATIONS[args.field_configuration]
        relation_class = RELATION_NAMES_TO_CLASSES[args.relation]
        trained_object_generator = object_gen.SmartBalancedBatchObjectGenerator(args.num_objects,
                                                                                field_configs, relation_class,
                                                                                max_recursion_depth=args.max_recursion_depth)

        test_object_generator = object_gen.SmartBalancedBatchObjectGenerator(args.model_test_value,
                                                                             field_configs, relation_class,
                                                                             max_recursion_depth=args.max_recursion_depth)

        train_dataset = datagen.ObjectGeneratorDataset(trained_object_generator, 0)
        # TODO: Does it make more sense to assign empty datasets than None?
        if args.validation_size is None:
            args.validation_size = args.dataset_size
        validation_dataset = datagen.ObjectGeneratorDataset(trained_object_generator, 0)

        if args.test_size is None:
            args.test_size = args.dataset_size
        test_dataset = datagen.ObjectGeneratorDataset(test_object_generator, args.test_size)


        model_configurations = MODEL_CONFIGURATIONS[args.model_configuration]

        for model_class, model_kwargs in model_configurations.items():
            model_class_name = prettify_class_name(model_class)
            if args.model is not None and len(args.model) > 0 and model_class_name not in args.model:
                print(f'Skipping model {model_class_name} because it is not in {args.model}')
                continue

            args.model_name = model_class_name

            model_kwargs['object_generator'] = test_object_generator
            model_kwargs['lr'] = args.learning_rate
            model_kwargs['batch_size'] = args.batch_size
            model_kwargs['train_epoch_size'] = args.dataset_size
            model_kwargs['validation_epoch_size'] = args.validation_size
            model_kwargs['regenerate_every_epoch'] = False
            model_kwargs['train_dataset'] = train_dataset
            model_kwargs['validation_dataset'] = validation_dataset
            model_kwargs['test_dataset'] = test_dataset
            model_kwargs['train_log_prefix'] = args.test_log_prefix
            model_kwargs['validation_log_prefix'] = args.test_log_prefix
            model_kwargs['test_log_prefix'] = args.test_log_prefix

            args.wandb_project = f'{args.relation}-relation-{args.model_configuration}-models-{args.num_objects}-objects-{args.dataset_size}-dataset'
            args.wandb_run_name = f'{model_class_name}-{args.seed}'

            print(f'Testing {args.wandb_project}/{args.wandb_run_name}, originally trained with {args.num_objects} objects, with {test_dataset.object_generator.n} test objects.')

            api = wandb.Api()
            runs = api.runs(f'{args.wandb_entity}/{args.wandb_project}', {'config.wandb_run_name': args.wandb_run_name})
            if len(runs) == 0:
                print(f'Failed to find run to resume for {args.wandb_run_name}. Moving to the next one... ')
                continue

            original_run = runs[0]
            os.environ['WANDB_RESUME'] = 'must'
            os.environ['WANDB_RUN_ID'] = original_run.id

            checkpoint_files = list(filter(lambda f: f.name.endswith('.ckpt'), original_run.files()))
            if len(checkpoint_files) == 0:
                print(f'Found no checkpoints for run {"/".join(original_run.path)}. Skipping... ')
                continue

            elif len(checkpoint_files) > 1:  # if more than one exists, take the latest
                epochs = [int([x for x in f.name.split('-') if x.startswith('epoch')][0].split('=')[1])
                          for f in checkpoint_files]
                checkpoint_file = checkpoint_files[np.argmax(epochs)]

            else:
                 checkpoint_file = checkpoint_files[0]

            os.makedirs(args.checkpoint_download_folder, exist_ok=True)
            checkpoint_file.download(replace=True, root=args.checkpoint_download_folder)
            checkpoint_path = os.path.join(args.checkpoint_download_folder, checkpoint_file.name)

            model = model_class.load_from_checkpoint(checkpoint_path, **model_kwargs)
            args.use_gpu = int(torch.cuda.is_available())

            logger = WandbLogger(args.wandb_run_name, args.wandb_dir, project=args.wandb_project,
                                 entity=args.wandb_entity, id=original_run.id, version=original_run.id)

            trainer = Trainer(logger=logger, gpus=args.use_gpu, max_epochs=1)
            trainer.test(model)

            logger.save()
            logger.close()

            del trainer
            del model


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

    main_args.test_field = main_args.test_field.replace('--', '').replace('-', '_')
    assert(main_args.test_field in main_args)

    main_arg_vars = vars(main_args)
    test_field_type = type(main_arg_vars[main_args.test_field])
    if test_field_type == list:
        test_field_type = type(main_arg_vars[main_args.test_field][0])
    main_args.test_values = [test_field_type(val) for val in main_args.test_values]

    multiple_option_fields = copy.copy(MULTIPLE_OPTION_REWRITE_FIELDS)
    assert (main_args.test_field in multiple_option_fields)
    multiple_option_fields.remove(main_args.test_field)

    multiple_option_field_values = [main_var_args[key] for key in multiple_option_fields]

    for value_combination in itertools.product(*multiple_option_field_values):
        args_copy = copy.deepcopy(main_args)
        var_args_copy = vars(args_copy)
        var_args_copy.update({key: value for key, value in zip(multiple_option_fields,
                                                               value_combination)})

        run_generalization_test_single_setting(args_copy)


if __name__ == '__main__':
    main()

