__version__ = '0.0.1.dev201912190+925416d'
git_version = '925416dd39e8469dbf50541ff2566694e0c95a73'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
