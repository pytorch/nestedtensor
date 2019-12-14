__version__ = '0.0.1.dev201912143+f576372'
git_version = 'f576372b54f808280193f936cd6079e3037e2303'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
