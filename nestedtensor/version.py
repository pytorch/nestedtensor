__version__ = '0.0.1.dev202042815+f6b89c6'
git_version = 'f6b89c61680b3567208c021bd9a525e8974aee45'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
