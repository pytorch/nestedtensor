__version__ = '0.0.1.dev202053019+c1875c5'
git_version = 'c1875c5541ce5ad4283b9d11d6920de91a3c8ba7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
