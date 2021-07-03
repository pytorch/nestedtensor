__version__ = '0.1.4+8e4372a'
git_version = '8e4372a7ba45e1b2e3e27163518fd57b936fda9d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
