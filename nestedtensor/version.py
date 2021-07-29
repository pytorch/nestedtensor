__version__ = '0.1.4+29f966d'
git_version = '29f966d8e60bf3912bf6598e9e325f4588f99c0e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
