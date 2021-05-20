__version__ = '0.1.4+7748fec'
git_version = '7748fec4761b901e40b91131f2a48372c8a093eb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
