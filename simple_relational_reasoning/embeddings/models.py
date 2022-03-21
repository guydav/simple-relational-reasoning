from collections import OrderedDict
import os
import torch
from torch import nn
import torchvision.models as models


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

def build_model(name, device, pretrained=True, saycam=None, flip=None, dino=None):
    name = name.lower()
    assert(name in MODELS)
    model = None
    
    if dino:
        assert(dino in DINO_OPTIONS)
        assert(name == RESNEXT)

        checkpoint_path = os.path.join(CHECKPOINT_FOLDER, f'DINO-{dino}.pth')
        model = models.resnext50_32x4d(pretrained=False)
        model = load_dino_model(model, checkpoint_path, False)   
        model = model.to(device)
        model.fc = nn.Sequential()

    elif flip:
        assert(flip in FLIPPING_OPTIONS)
        assert(name == RESNEXT)

        checkpoint = torch.load(os.path.join(CHECKPOINT_FOLDER, f'TC-S-{flip}.tar'))

        model = models.resnext50_32x4d(pretrained=False)
        model = nn.DataParallel(model) 
        model = model.to(device)
        model.module.fc = nn.Linear(2048, FLIPPING_n_out)
        # TODO: if this fails, model might have been saved from cpu, should mmove to device later
        model.load_state_dict(checkpoint['model_state_dict'])
        model.module.fc = nn.Sequential()

    elif saycam:
#         print('Loading SAYcam model')
        if saycam is True:
            saycam = 'SAY'
        saycam = saycam.upper()
        
        assert(saycam in SAYCAM_n_out)
        assert(name in SAYCAM_models)
        
        checkpoint = torch.load(os.path.join(CHECKPOINT_FOLDER, f'TC-{saycam}-{name}.tar'))
        
        if name == MOBILENET:
            model = models.mobilenet_v2(pretrained=False)
            model = nn.DataParallel(model)
            model = model.to(device)
            model.module.classifier = nn.Linear(1280, SAYCAM_n_out[saycam])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.module.classifier = nn.Sequential()
        
        elif name == RESNEXT:
            model = models.resnext50_32x4d(pretrained=False)
            model = nn.DataParallel(model)
            model = model.to(device)
            model.module.fc = nn.Linear(2048, SAYCAM_n_out[saycam])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.module.fc = nn.Sequential()
    
    else:
#         print('Loading ImageNet or random model')
        if name == RESNET:
            model = models.resnet18(pretrained=pretrained)
            model.fc_backup = model.fc
            model.fc = nn.Sequential()
            model = model.to(device)

        elif name == VGG:
            model = models.vgg16(pretrained=pretrained)
            model.fc_backup = model.classifier[6]
            model.classifier[6] = nn.Sequential()
            model = model.to(device)

        elif name == MOBILENET:
            model = models.mobilenet_v2(pretrained=pretrained)
            model.fc_backup = model.classifier[1]
            model.classifier[1] = nn.Sequential()
            model = model.to(device)

        elif name == RESNEXT:
            model = models.resnext50_32x4d(pretrained=pretrained)
            model.fc_backup = model.fc
            model.fc = nn.Sequential()
            model = model.to(device)
        
    if model is None:
        raise ValueError(f'Failed to build model for name={name}, pretrained={pretrained}, saycam={saycam}')
       
#     print(name, pretrained, saycam, type(model))
        
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
    