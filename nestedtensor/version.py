__version__ = '0.0.1.dev202071619+6fdf4ce'
git_version = '6fdf4ce1d3959ffe9135ec469bc0a891617536ce'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
