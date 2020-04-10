__version__ = '0.0.1.dev202041018+103126f'
git_version = '103126f48c7255ef4aec863e4d9f9ba68e5f933e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
