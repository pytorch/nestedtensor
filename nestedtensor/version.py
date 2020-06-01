__version__ = '0.0.1.dev2020614+2da846f'
git_version = '2da846ff4e4d3fa2705e9245b94220fb7262b43b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
