__version__ = '0.0.1.dev202012020+7a5cb4c'
git_version = '7a5cb4c053f91db4b5665e1966df26896f5dd735'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
