__version__ = '0.0.1.dev202071518+c2bbfa6'
git_version = 'c2bbfa6222d6aa71c1bdf548e691d5556b26ab30'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
