__version__ = '0.0.1.dev202081921+b5cb521'
git_version = 'b5cb521993070d7c8cb7b9dd4c83574a8e88641c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
