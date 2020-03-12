__version__ = '0.0.1.dev202031019+2791085'
git_version = '27910851d4e9f07edef787a41eb5ff3bd22d1c07'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
