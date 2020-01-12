__version__ = '0.0.1.dev20201122+7f14ac1'
git_version = '7f14ac17cec7453607dbfa9534e1bd45f60aba35'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
