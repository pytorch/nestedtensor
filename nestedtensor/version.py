__version__ = '0.0.1+f37d43d'
git_version = 'f37d43d40654ced322b716d3b8f3b7fd5da5ff7e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
