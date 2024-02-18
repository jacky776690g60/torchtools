"""
torchtools contains all the useful functions I found and developed throughout 
all projects as time goes on.
"""

from .painters import DaVinci, Mondrian 
from .imagefolder import IMG_EXTENSIONS, ImageFolder
from .gimmick import Optics, save_model, set_seeds
from .utility import *

"""
If __all__ is defined: Only the names included in the __all__ list will be 
imported when a client uses the `from module import * syntax`.
"""
__all__ = [
    utility.__all__
]

"""
If __all__ is not defined: All names that do not begin with an underscore will 
be imported when a client uses the from module import * syntax.
"""