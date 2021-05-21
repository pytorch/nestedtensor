__version__ = '0.1.4+7ee35c9'
git_version = '7ee35c9618fdb3e876b338414c22ea864be90676'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
