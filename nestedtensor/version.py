__version__ = '0.0.1.dev20201144+566019c'
git_version = '566019cd5692368c095f9c184a307a41d0dc18ba'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
