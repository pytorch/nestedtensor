__version__ = '0.1.4+5ab1be1'
git_version = '5ab1be1385ab8c35f4183145736485f3c887b479'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
