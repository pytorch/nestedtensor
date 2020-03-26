__version__ = '0.0.1.dev20203263+7838319'
git_version = '7838319ade9e95c13da8e103101a9afab0ee9679'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
