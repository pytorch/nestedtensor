__version__ = '0.1.2'
git_version = '1be03c377a2cc3c56be354843cb2a48a1e422b24'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
