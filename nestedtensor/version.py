__version__ = '0.0.1.dev20208619+7846ae0'
git_version = '7846ae032212178af43579c7c31b3315a5dbd750'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
