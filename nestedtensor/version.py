__version__ = '0.1.4+81c990e'
git_version = '81c990e7cc2e7c647f7205cb587392bf3e61f7f3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
