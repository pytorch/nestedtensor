__version__ = '0.0.1.dev20191280+4015971'
git_version = '40159719813d03dcdf46cd34a99f100d845772f3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
