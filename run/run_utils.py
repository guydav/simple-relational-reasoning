import argparse
import os
import sys
import typing

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

from simple_relational_reasoning.embeddings.models import BASELINE_MODELS, FLIPPING_MODELS, DINO_MODELS


def args_to_model_configurations(args: argparse.Namespace) -> typing.Tuple[typing.List[typing.Dict[str, typing.Any]], typing.List[str]]:
    model_kwarg_dicts = []
    model_names = []

    for model_name in args.model:
        if model_name in BASELINE_MODELS and args.saycam:
            model_kwarg_dicts.append(dict(model_name=model_name, device=args.device, pretrained=False, saycam=args.saycam, unpooled_output=args.unpooled_output))
            model_names.append(f'{model_name}-saycam({args.saycam})')
        
        if model_name in BASELINE_MODELS and args.imagenet:
            model_kwarg_dicts.append(dict(model_name=model_name, device=args.device, pretrained=True, unpooled_output=args.unpooled_output))
            model_names.append(f'{model_name}-imagenet')

        if model_name in BASELINE_MODELS and args.untrained:
            model_kwarg_dicts.append(dict(model_name=model_name, device=args.device, pretrained=False, unpooled_output=args.unpooled_output))
            model_names.append(f'{model_name}-random')

        if model_name in FLIPPING_MODELS and args.flipping and len(args.flipping) > 0:
            for flip_type in args.flipping:
                model_kwarg_dicts.append(dict(model_name=model_name, device=args.device, 
                    pretrained=False, flip=flip_type, unpooled_output=args.unpooled_output))

                model_names.append(f'{model_name}-saycam(S)-{flip_type}')

        if model_name in DINO_MODELS and args.dino and len(args.dino) > 0:
            for dino in args.dino:
                model_kwarg_dicts.append(dict(model_name=model_name, device=args.device, 
                    pretrained=False, dino=dino, unpooled_output=args.unpooled_output))

                model_names.append(f'{model_name}-DINO-{dino}')

    return model_kwarg_dicts, model_names
