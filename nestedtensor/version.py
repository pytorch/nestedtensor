__version__ = '0.0.1+7f42e8a'
git_version = '7f42e8a0ba8a7f79d8e89665e15c38af2dd454a9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
