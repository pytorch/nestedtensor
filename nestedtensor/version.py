__version__ = '0.0.1.dev20204274+0a446e8'
git_version = '0a446e8780c294cb156693e0b434b736a37b84d7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
