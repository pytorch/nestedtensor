__version__ = '0.0.1.dev20206123+705f0be'
git_version = '705f0be14de63f9b611bc77c61e3a54a12863660'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
