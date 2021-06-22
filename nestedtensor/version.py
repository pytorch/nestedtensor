__version__ = '0.1.4+60babfa'
git_version = '60babfa2073a20de43e262e62602dd0e8d365cd0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
