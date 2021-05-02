__version__ = '0.1.4+9aa542c'
git_version = '9aa542cc6c333ae0b15a1df1ddef31b61cf57358'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
