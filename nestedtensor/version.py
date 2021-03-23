__version__ = '0.0.1+39e71e6'
git_version = '39e71e64b945fca8dd14f791a2fa40cbdb9d85e4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
