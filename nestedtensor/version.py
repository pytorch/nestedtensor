__version__ = '0.1.4+7136014'
git_version = '7136014de816f73e9a6c9218e2ff13891932f6cd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
