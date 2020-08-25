__version__ = '0.0.1.dev202082520+3c1533e'
git_version = '3c1533eb097be8c70154a21d844667d3a4017f0f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
