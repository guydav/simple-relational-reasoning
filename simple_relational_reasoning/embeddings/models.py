from collections import OrderedDict
import os
import torch
from torch import nn
import torchvision.models as models
import typing


CHECKPOINT_FOLDER = r'/home/gd1279/scratch/SAYcam-models'

RESNET = 'resnet'
VGG = 'vgg'
MOBILENET = 'mobilenet'
RESNEXT = 'resnext'
MODELS = (RESNET, VGG, MOBILENET, RESNEXT)

SAYCAM_models = (MOBILENET, RESNEXT)
SAYCAM_n_out = {'S': 2765, 'SAY':6269}
FLIPPING_n_out = 2575

FLIPPING_OPTIONS = ('s', 'h', 'v', 'hv')
DINO_OPTIONS = ('S', 'ImageNet')


MODEL_EMBEDDING_DIMENSIONS = {
    (MOBILENET, True): 62720,
    (MOBILENET, False): 1280,
    (RESNEXT, True): 100352,
    (RESNEXT, False): 2048,
}


class MobileNetV2UnpooledWrapper(nn.Module):
    def __init__(self, mobilenet_model: nn.Module):
        super(MobileNetV2UnpooledWrapper, self).__init__()
        self.mobilenet_model = mobilenet_model
        self.embedding_dim = mobilenet_model.embedding_dim

    def forward(self, x):
        if isinstance(self.mobilenet_model, nn.DataParallel):
            return self.mobilenet_model.module.features(x)  # type: ignore
        else:
            return self.mobilenet_model.features(x)  # type: ignore


def build_model(name: str, device: str, pretrained: bool = True, saycam: typing.Union[None, bool, str] = None, flip: typing.Optional[str] = None, 
    dino: typing.Optional[str] = None, unpooled_output: bool = False):
    name = name.lower()
    assert(name in MODELS)
    model = None
    
    if dino:
        assert(dino in DINO_OPTIONS)
        assert(name == RESNEXT)

        checkpoint_path = os.path.join(CHECKPOINT_FOLDER, f'DINO-{dino}.pth')
        model = models.resnext50_32x4d(weights=None)
        model = load_dino_model(model, checkpoint_path, False)   
        model = model.to(device)
        model.embedding_dim = MODEL_EMBEDDING_DIMENSIONS[(name, unpooled_output)]  # type: ignore
        model.fc = nn.Identity()  # type: ignore
        if unpooled_output:
            model.avgpool = nn.Identity()  # type: ignore

    elif flip:
        assert(flip in FLIPPING_OPTIONS)
        assert(name == RESNEXT)

        checkpoint = torch.load(os.path.join(CHECKPOINT_FOLDER, f'TC-S-{flip}.tar'))

        model = models.resnext50_32x4d(weights=None)
        model = nn.DataParallel(model)   # type: ignore
        model = model.to(device)
        model.module.fc = nn.Linear(2048, FLIPPING_n_out)
        # TODO: if this fails, model might have been saved from cpu, should mmove to device later
        model.load_state_dict(checkpoint['model_state_dict'])
        model.embedding_dim = MODEL_EMBEDDING_DIMENSIONS[(name, unpooled_output)]  # type: ignore
        model.module.fc = nn.Identity()  # type: ignore
        if unpooled_output:
            model.module.avgpool = nn.Identity()  # type: ignore

    elif saycam:
#         print('Loading SAYcam model')
        if saycam is True:
            saycam = 'SAY'
        saycam = saycam.upper()
        
        assert(saycam in SAYCAM_n_out)
        assert(name in SAYCAM_models)
        
        checkpoint = torch.load(os.path.join(CHECKPOINT_FOLDER, f'TC-{saycam}-{name}.tar'))
        
        if name == MOBILENET:
            model = models.mobilenet_v2(weights=None)
            model = nn.DataParallel(model)  # type: ignore
            model = model.to(device)
            model.module.classifier = nn.Linear(1280, SAYCAM_n_out[saycam])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.embedding_dim = MODEL_EMBEDDING_DIMENSIONS[(name, unpooled_output)]  # type: ignore
            model.module.classifier = nn.Identity()  # type: ignore
            if unpooled_output:
                model = MobileNetV2UnpooledWrapper(model)

        
        elif name == RESNEXT:
            model = models.resnext50_32x4d(weights=None)
            model = nn.DataParallel(model)  # type: ignore
            model = model.to(device)
            model.module.fc = nn.Linear(2048, SAYCAM_n_out[saycam])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.embedding_dim = model.module.fc.in_features  # type: ignore
            model.embedding_dim = MODEL_EMBEDDING_DIMENSIONS[(name, unpooled_output)]  # type: ignore
            model.module.fc = nn.Identity()  # type: ignore
            if unpooled_output:
                model.module.avgpool = nn.Identity()  # type: ignore
    
    else:
#         print('Loading ImageNet or random model')
        # if name == RESNET:
        #     model = models.resnet18(pretrained=pretrained)
        #     model.fc_backup = model.fc
        #     model.embedding_dim = model.fc.in_features  # type: ignore
        #     model.fc = nn.Identity()  # type: ignore
        #     model = model.to(device)

        # elif name == VGG:
        #     model = models.vgg16(pretrained=pretrained)
        #     model.fc_backup = model.classifier[6]
        #     model.embedding_dim = model.classifier[6].in_features  # type: ignore
        #     model.classifier[6] = nn.Identity()
        #     model = model.to(device)
        # elif name == MOBILENET:
        if name == MOBILENET:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.mobilenet_v2(weights=weights)
            model.fc_backup = model.classifier[1]
            model.embedding_dim = MODEL_EMBEDDING_DIMENSIONS[(name, unpooled_output)]  # type: ignore
            model.classifier = nn.Identity()  # type: ignore
            model = model.to(device)
            if unpooled_output:
                model = MobileNetV2UnpooledWrapper(model)

        elif name == RESNEXT:
            weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnext50_32x4d(weights=weights)
            model.fc_backup = model.fc
            model.embedding_dim = MODEL_EMBEDDING_DIMENSIONS[(name, unpooled_output)]  # type: ignore
            model.module.fc = nn.Identity()  # type: ignore
            if unpooled_output:
                model.module.avgpool = nn.Identity()  # type: ignore
            model = model.to(device)
        
    if model is None:
        raise ValueError(f'Failed to build model for name={name}, pretrained={pretrained}, saycam={saycam}')
        
    return model


def load_dino_model(model, checkpoint_path, verbose=False):
    """
    Args:
        model (a torchvision model): Initial model to be filled.
        checkpoint_path (path): path where the pretrained model checkpoint is stored.
    Returns:
        the filled model.
    """
    checkpoint = torch.load(checkpoint_path)
    student_state_dict = checkpoint['student']
    new_student_state_dict = OrderedDict()

    if verbose:
        print('=== Initial model state dict keys ===')
        print(model.state_dict().keys())

    for key in model.state_dict().keys():
        
        if 'module.backbone.' + key in student_state_dict.keys():
            new_student_state_dict[key] = student_state_dict['module.backbone.' + key]
            if verbose:
                print('Parameter', key, 'taken from the pretrained model')
        else:    
            new_student_state_dict[key] = model.state_dict()[key]
            if verbose:
                print('Parameter', key, 'taken from the random init')

    model.load_state_dict(new_student_state_dict)

    return model
    