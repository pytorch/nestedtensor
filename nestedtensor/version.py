__version__ = '0.0.1.dev202052020+fed9991'
git_version = 'fed99911848c4c30fa6a2fc85f012f5eb4587699'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
