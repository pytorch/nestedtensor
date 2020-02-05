__version__ = '0.0.1.dev2020254+48db4de'
git_version = '48db4de3146139e4dcc659e242ec01f2a883a5ed'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
