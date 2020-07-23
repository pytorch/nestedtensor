__version__ = '0.0.1.dev202072320+1afe9bf'
git_version = '1afe9bfcb860fad00e53be1b1ddf3bff9aaf05e8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
