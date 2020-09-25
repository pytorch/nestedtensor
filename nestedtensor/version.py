__version__ = '0.0.1.dev20209233+ce52c68'
git_version = 'ce52c68d5b284869956e74c2ee3aef028d44e455'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
