__version__ = '0.0.1.dev20208203+5f8b907'
git_version = '5f8b9073e2ba26762d5cdc8ad9ef5c419f41a415'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
