__version__ = '0.1.4+c575087'
git_version = 'c575087f2a6b8906dbd845b3880f88ee10955d83'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
