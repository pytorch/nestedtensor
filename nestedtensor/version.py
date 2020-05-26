__version__ = '0.0.1.dev202052621+174be9e'
git_version = '174be9e656ecb9921461b92fb438856b3488a5ab'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
