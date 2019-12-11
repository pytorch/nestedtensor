__version__ = '0.0.1.dev2019121019+f2c3cb6'
git_version = 'f2c3cb64575e8601fe6dee58f84373dabb65ead3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
