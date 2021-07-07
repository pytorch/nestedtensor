__version__ = '0.1.4+132f191'
git_version = '132f19170b92be10de62391052dc78a90fb2f4bd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
