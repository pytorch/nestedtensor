__version__ = '0.1.4+bc92f95'
git_version = 'bc92f953cc0d79ba31a03d6b54ffb03b99254a5d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
