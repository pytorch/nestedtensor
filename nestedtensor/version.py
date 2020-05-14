__version__ = '0.0.1.dev202051421+2b76b79'
git_version = '2b76b7950bea267545c7e2209c17d7c4da03e23c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
