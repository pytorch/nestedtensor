__version__ = '0.1.4+93c60af'
git_version = '93c60afca0d95337e30d171951b579211ddfff19'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
