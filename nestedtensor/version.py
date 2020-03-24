__version__ = '0.0.1.dev202032418+ccc2e07'
git_version = 'ccc2e07f013a0a4d378bfd35e08143c25ce89268'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
