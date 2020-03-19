__version__ = '0.0.1.dev202031914+1e7e306'
git_version = '1e7e3060a9d69e3710eb1380e98f7e55b6fc0aa1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
