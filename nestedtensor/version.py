__version__ = '0.0.1.dev202042418+9b2a76a'
git_version = '9b2a76a3c2ef8e5aad5e32e88189c54adadf2436'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
