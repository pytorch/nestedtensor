__version__ = '0.1.4+29a8f74'
git_version = '29a8f74d5700e04828cc47211046b63be09a16f7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
