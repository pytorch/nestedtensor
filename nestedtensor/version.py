__version__ = '0.0.1.dev202061717+6384059'
git_version = '63840592e8e5a3841fdc02e01aefa90e5f8031e9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
