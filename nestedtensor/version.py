__version__ = '0.1.4+49f85e2'
git_version = '49f85e277056264fb666e6de64e3b26c56af4ffc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
