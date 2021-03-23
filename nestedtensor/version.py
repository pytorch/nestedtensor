__version__ = '0.0.1+78a8606'
git_version = '78a8606575efeae8fface944d95926246c9d1e4e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
