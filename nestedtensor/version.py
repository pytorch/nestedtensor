__version__ = '0.0.1.dev202082121+a8ad5f1'
git_version = 'a8ad5f15a8025ad4e695c301647cff90c60c218e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
