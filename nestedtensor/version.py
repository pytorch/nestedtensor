__version__ = '0.1.4+54b8538'
git_version = '54b853835af76286ee6c2ea05918c05e5da2828f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
