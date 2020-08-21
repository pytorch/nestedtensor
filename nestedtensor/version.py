__version__ = '0.0.1.dev20208212+89aa737'
git_version = '89aa737c22b5789942a9ae5d0e98eb40dfb198cb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
