__version__ = '0.0.1.dev202011101+8b588d0'
git_version = '8b588d09ecff8aae07c85fdce0b32873f55d277b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
