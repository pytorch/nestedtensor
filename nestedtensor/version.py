__version__ = '0.1.4+08d8d60'
git_version = '08d8d6021e289990242bc1337e1e81a53e20bd4d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
