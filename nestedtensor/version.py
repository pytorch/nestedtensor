__version__ = '0.0.1+805ac22'
git_version = '805ac226590d1cc8c634c2b9a904259631dd825c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
