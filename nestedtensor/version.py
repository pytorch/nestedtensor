__version__ = '0.1.4+c6d13f9'
git_version = 'c6d13f91a24fb7ef4352777f9bcd76f100cbca37'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
