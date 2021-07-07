__version__ = '0.1.4+6bea731'
git_version = '6bea7318c6e1eed1e8ce99f87daa2c3994311251'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
