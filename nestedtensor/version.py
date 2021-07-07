__version__ = '0.1.4+9e9b51f'
git_version = '9e9b51fb5f3df1d8d3f227159bb64b0fc69fb3c8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
