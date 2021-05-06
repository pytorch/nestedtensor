__version__ = '0.1.4+0e3cf12'
git_version = '0e3cf126acdf7e2b70f17f69afdae28230c5d3fd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
