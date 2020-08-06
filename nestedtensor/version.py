__version__ = '0.0.1.dev20208623+500a9f1'
git_version = '500a9f11737cd5fb4b9f9976b46ce14781c196e2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
