__version__ = '0.0.1.dev202053121+c58d534'
git_version = 'c58d5347880369b639cf941cf4dc31c6d2b8fbbb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
