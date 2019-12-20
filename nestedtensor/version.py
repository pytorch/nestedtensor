__version__ = '0.0.1.dev201912202+eee010d'
git_version = 'eee010ddbec748509e3aeeaa10d545d1f50bac5c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
