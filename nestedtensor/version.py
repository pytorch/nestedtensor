__version__ = '0.1.4+4f87969'
git_version = '4f8796980571f95f8e5a960c79bf2abaab73c856'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
