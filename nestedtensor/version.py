__version__ = '0.1.4+52b805a'
git_version = '52b805adc622654260fe3640fa3b7c058b4c9e04'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
