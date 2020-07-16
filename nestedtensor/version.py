__version__ = '0.0.1.dev20207162+0b7a551'
git_version = '0b7a551cb6b0779e2a0e7720cfcd69aa893e67f7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
