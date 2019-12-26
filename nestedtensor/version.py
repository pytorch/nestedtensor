__version__ = '0.0.1.dev2019122618+2ba36d2'
git_version = '2ba36d2a230e381a885fb1289fd773f67c235fcf'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
