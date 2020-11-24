__version__ = '0.0.1.dev2020112423+d81e8c3'
git_version = 'd81e8c336263ba813e97b174cb6d2a4cdd858a6d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
