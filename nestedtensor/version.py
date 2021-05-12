__version__ = '0.1.4+187bcdd'
git_version = '187bcdde7fdd4c6d4ceb951ab862536322a853ca'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
