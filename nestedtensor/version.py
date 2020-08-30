__version__ = '0.0.1.dev202083018+24b10d6'
git_version = '24b10d6c87d60aa998c5110c996628e41a2687fa'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
