__version__ = '0.0.1.dev201912916+6367f27'
git_version = '6367f272b1bb37791dceaf015272fc6aacef0261'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
