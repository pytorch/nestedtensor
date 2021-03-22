__version__ = '0.0.1+ea1aa01'
git_version = 'ea1aa015aa7bf4c6e515234d1288351ba9bed57f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
USE_SUBMODULE=False
