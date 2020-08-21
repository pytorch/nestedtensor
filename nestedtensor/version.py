__version__ = '0.0.1.dev20208213+9a40ce5'
git_version = '9a40ce5edafda1f0eb307181c0202191c63bcb01'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
