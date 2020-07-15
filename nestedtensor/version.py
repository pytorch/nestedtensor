__version__ = '0.0.1.dev202071521+092c1e4'
git_version = '092c1e44a8f62d3aa6904e0780e4bdf94adeaa90'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
