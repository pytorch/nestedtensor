__version__ = '0.0.1.dev20208182+edc64d8'
git_version = 'edc64d8d100cc9b1a945428045f062dbbe6a39ad'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
