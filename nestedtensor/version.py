__version__ = '0.1.4+1e9cb12'
git_version = '1e9cb12065b38ac43008d319e82166b66af7d30a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
