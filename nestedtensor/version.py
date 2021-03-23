__version__ = '0.0.1+7c3f1f7'
git_version = '7c3f1f7986ef7ff9356a4e260751ed6cbbb4d2d5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
