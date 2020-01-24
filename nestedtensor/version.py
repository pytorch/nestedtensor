__version__ = '0.0.1.dev202012421+991c1ba'
git_version = '991c1ba3665c8e688b25cb8df380c51efc7ae590'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
