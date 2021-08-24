__version__ = '0.1.4+f4ef725'
git_version = 'f4ef7252179683b274d6c9077db9dade2d178626'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
