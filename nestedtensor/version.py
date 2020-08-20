__version__ = '0.0.1.dev202082022+5ecc929'
git_version = '5ecc929fa4d801ba55046321a4796d83470bde07'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
