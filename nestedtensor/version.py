__version__ = '0.1.4+7f450ef'
git_version = '7f450ef423d8392c43ef842dccb8a34e4ad9d7f7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
