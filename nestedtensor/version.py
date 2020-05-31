__version__ = '0.0.1.dev202053117+c1c8184'
git_version = 'c1c8184de1624bf4f9ac1bb2943b8cfe797b866d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
