__version__ = '0.1.4+04095b1'
git_version = '04095b1f776a4fcf55f6869c7128d4e6bd8fe20f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
