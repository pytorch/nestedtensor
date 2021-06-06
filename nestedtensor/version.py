__version__ = '0.1.4+63b16e9'
git_version = '63b16e926593420842120daff724656c14cb1b3c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
