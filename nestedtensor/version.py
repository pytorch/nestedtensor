__version__ = '0.0.1.dev202081416+bafd896'
git_version = 'bafd89620a11c3fdf7d561d415d1303786f5f2e4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
