__version__ = '0.0.1.dev2019121121+a4d5d6c'
git_version = 'a4d5d6cfbe1fcf4fdb8f871d24f89e86ffb22aab'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
