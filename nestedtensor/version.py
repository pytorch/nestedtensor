__version__ = '0.1.4+217f4ad'
git_version = '217f4adf6f4b89967779c0a20585c4454da17a82'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
