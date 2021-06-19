__version__ = '0.1.4+31bb78f'
git_version = '31bb78f8358d9f418c8d91bf8e7d5a6d07349448'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
