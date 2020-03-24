__version__ = '0.0.1.dev202032419+eed109f'
git_version = 'eed109fa6c2c6bfcb1124348edaffa20f072180f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
