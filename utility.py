"torchtools utility"

import os, sys
from typing import NewType
import PIL.Image

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torchvision

__all__ = (
    'DEVICE',
    # Types
    'npNDArrayType', 'PILImageType', 'PILImageResamplingType', 'TorchTensorType', 
    'TorchDatasetType', 'TorchTransformsComposeType', 'TorchDataLoaderType', 
    'TorchNNModuleType', 'TorchOptimizerType', 'TorchLossFunctionType', 'PltFigureType',
    # Fonts
    'FONT_TITLE_BG', 'FONT_TITLE_MD', 'FONT_TITLE_SM', 'FONT_SUBTITLE_MD', 
    'FONT_SUBTITLE_SM', 'FONT_CONTEXT_0'
)

# =============================
# Global Variables
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
"Global device for torch"



# =============================
# Types
# =============================
npNDArrayType              = NewType("np.ndarray", np.ndarray)

PILImageType               = NewType("PIL.Image.Image", PIL.Image.Image)
PILImageResamplingType     = NewType("PIL.Image.Resampling", PIL.Image.Resampling)

TorchTensorType            = NewType("torch.Tensor", torch.Tensor)
TorchDatasetType           = NewType("torch.utils.data.Dataset", torch.utils.data.Dataset)
TorchTransformsComposeType = NewType("torchvision.transforms.Compose", torchvision.transforms.Compose)
TorchDataLoaderType        = NewType("torch.utils.data.DataLoader", torch.utils.data.DataLoader)
TorchNNModuleType          = NewType("torch.nn.Module", torch.nn.Module)
TorchOptimizerType         = NewType("torch.optim.Optimizer", torch.optim.Optimizer)
TorchLossFunctionType      = NewType("torch.nn.modules.loss._Loss", torch.nn.modules.loss._Loss)

PltFigureType              = NewType("plt.Figure", plt.Figure)




# =============================
# Classes
# =============================
class Fontbook:
    '''
    Class to manage font styles for `matplotlib` visualizations.
    '''
    def __init__(self, 
        fontname      = 'sans-serif', 
        fontsize      = 12, 
        fontweight    = 'normal', 
        color         = 'black'
    ):
        self._font_settings = {
            'fontname':     fontname,
            'fontsize':     fontsize,
            'fontweight':   fontweight,
            'color':        color
        }

    def build(self) -> dict:
        '''
        Build and return the font dictionary to be used with Matplotlib.
        '''
        return self._font_settings.copy()

    def update(self, **kwargs):
        '''
        Update font properties.
        '''
        self._font_settings.update(kwargs)

# Global font settings
_fontbook = Fontbook(fontname='Arial', fontsize=14, fontweight='bold', color='black')
FONT_TITLE_BG       = _fontbook.build()
'Font to be used in `matplotlib`'

_fontbook.update(fontsize=12)
FONT_TITLE_MD       = _fontbook.build()
'Font to be used in `matplotlib`'

_fontbook.update(fontsize=10)
FONT_TITLE_SM       = _fontbook.build()
'Font to be used in `matplotlib`'

_fontbook.update(fontsize=12, fontweight='normal')
FONT_SUBTITLE_MD    = _fontbook.build()
'Font to be used in `matplotlib`'

_fontbook.update(fontsize=10)
FONT_SUBTITLE_SM    = _fontbook.build()
'Font to be used in `matplotlib`'

_fontbook.update(fontname='Verdana', fontsize=11)
FONT_CONTEXT_0      = _fontbook.build()
'Font to be used in `matplotlib`'