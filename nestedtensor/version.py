__version__ = '0.0.1.dev202071021+1b3e405'
git_version = '1b3e4052bd551034858b2878573c1f0e6da32119'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
