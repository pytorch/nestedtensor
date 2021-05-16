__version__ = '0.1.4+3be6cae'
git_version = '3be6caee9833d022eaf40c7a80e8ef3d3d2190a0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
