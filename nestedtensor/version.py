__version__ = '0.1.4+f633993'
git_version = 'f633993c4a0bd6588d998cccbb72bacc312ef47b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
