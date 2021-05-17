__version__ = '0.1.4+fbdd335'
git_version = 'fbdd335e410c7b3cf7970fbd65db181e9302e07d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
