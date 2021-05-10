__version__ = '0.1.4+fe522e4'
git_version = 'fe522e4e2b9a1ca25297b001f1d2e8083c829b1f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
