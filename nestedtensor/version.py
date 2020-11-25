__version__ = '0.0.1.dev2020112523+2f288bc'
git_version = '2f288bcd04a36b3dd24d7a59eed5440e5a92a675'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
