__version__ = '0.0.1.dev2020936+e2d04a8'
git_version = 'e2d04a813dbc7132c0a837251031e844b2be105b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
