__version__ = '0.1.4+c5d6f3d'
git_version = 'c5d6f3d6764c2c0d214eca2ef8df79171c4e9447'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
