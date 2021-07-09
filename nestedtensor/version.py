__version__ = '0.1.4+0c811ed'
git_version = '0c811ed766b8190728a36200b8bcbf2319714d24'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
