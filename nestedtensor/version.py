__version__ = '0.1.4+df82e9d'
git_version = 'df82e9dfc6a83e3a6415e37ce92c664337d0e117'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
