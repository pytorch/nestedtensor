__version__ = '0.1.4+6a7262b'
git_version = '6a7262ba72a9b2db656fcf1db2c70e14d4769f07'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
