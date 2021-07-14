__version__ = '0.1.4+28b8da8'
git_version = '28b8da861a47878a98ac270f733e7e0e075b7ae4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
