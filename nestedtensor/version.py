__version__ = '0.0.1+5dde999'
git_version = '5dde999f79737000b6048dce126967661e23ad34'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
