__version__ = '0.1.4+cd74160'
git_version = 'cd741600503ab03a967aee1a45d995dcfa7f5c3c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
