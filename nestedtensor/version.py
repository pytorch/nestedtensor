__version__ = '0.0.1.dev202083019+7a11d31'
git_version = '7a11d31cc78b7d76678688b4894d01d3c9358743'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
