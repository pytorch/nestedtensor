__version__ = '0.1.4+557edb6'
git_version = '557edb6fb85b438bcccba1ab3f1c39a4a8d13961'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
