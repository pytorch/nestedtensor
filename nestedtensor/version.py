__version__ = '0.1.4+3f32b9d'
git_version = '3f32b9d226af4b50546e8952c06a8ea4760bee2e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
