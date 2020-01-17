__version__ = '0.0.1.dev202011716+bc96e2e'
git_version = 'bc96e2e33c2fafd29bca8521d811849ecc673f76'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
