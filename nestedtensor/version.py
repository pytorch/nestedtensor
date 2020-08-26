__version__ = '0.0.1.dev202082621+e8785f3'
git_version = 'e8785f35c95ff4eefd88d62c8583621a952edb4c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
