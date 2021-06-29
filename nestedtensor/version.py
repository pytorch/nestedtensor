__version__ = '0.1.4+54117b5'
git_version = '54117b58363575602eab13131808959482a13511'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
