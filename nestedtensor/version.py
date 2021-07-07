__version__ = '0.1.4+6a0ba02'
git_version = '6a0ba024f487bfc31492e25e2227cd209b145e39'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
