__version__ = '0.0.1.dev2020365+618cf5e'
git_version = '618cf5e7e095018d4003749a39adeed1792b7bb7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
