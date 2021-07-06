__version__ = '0.1.4+f6a482f'
git_version = 'f6a482fa97f6c958f2911884292cc886e85d6214'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
