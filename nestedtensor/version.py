__version__ = '0.1.4+dd2f8f0'
git_version = 'dd2f8f0fb92ce3f93a6448dfc50d18bf66be9b83'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
