__version__ = '0.0.1.dev2019121619+221a7fc'
git_version = '221a7fcd8678cc470ed5f71c60ec18ee43a53437'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
