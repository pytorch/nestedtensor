__version__ = '0.0.1.dev2020663+3ec50e8'
git_version = '3ec50e8b9abd9d98e5b8ad972ff9bcc1325acf15'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
