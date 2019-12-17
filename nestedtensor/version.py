__version__ = '0.0.1.dev201912174+ee5f59f'
git_version = 'ee5f59f8f7311b5a99a956e0fc3de9a37e3f7c19'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
