__version__ = '0.0.1.dev2020112422+e1b99ab'
git_version = 'e1b99ab3a5d2bd299977b75ba03bbfdc62728ad1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
