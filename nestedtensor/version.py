__version__ = '0.0.1.dev202053116+90eb2df'
git_version = '90eb2dfe479ff53f4d4c0a72154a7b6f413b4b65'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
