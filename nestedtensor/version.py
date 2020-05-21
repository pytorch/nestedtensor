__version__ = '0.0.1.dev202052023+2de5c5f'
git_version = '2de5c5fb95e5fdc06d1d38ba14bf47d2d31c8f7e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
