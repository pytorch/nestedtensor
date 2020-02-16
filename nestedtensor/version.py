__version__ = '0.0.1.dev202021518+3fc2c37'
git_version = '3fc2c370fa25bc61c69558e209331eeef2c48b15'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
