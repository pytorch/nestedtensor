__version__ = '0.0.1.dev20208216+2c5fa87'
git_version = '2c5fa873e88345937e8bb830678a788327a17958'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
