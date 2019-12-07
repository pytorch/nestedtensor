__version__ = '0.0.1.dev201912718+f1ff4c1'
git_version = 'f1ff4c16de03b19519dbd8ac2aa5594016913c39'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
