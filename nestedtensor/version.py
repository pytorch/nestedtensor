__version__ = '0.0.1.dev202082621+8eabb5b'
git_version = '8eabb5bc93b06c8a87c0c858347f7866d92149c4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
