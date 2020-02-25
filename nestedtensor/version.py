__version__ = '0.0.1.dev20202254+96ff8e5'
git_version = '96ff8e5dde852ad18feea566c7537fbb2c94139b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
