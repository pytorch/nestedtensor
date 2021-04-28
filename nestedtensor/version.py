__version__ = '0.0.1+af75937'
git_version = 'af7593774ae1a2fd0a9ce863e3b29765bfb1d498'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
