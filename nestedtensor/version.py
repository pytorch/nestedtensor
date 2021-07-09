__version__ = '0.1.4+b6dc27d'
git_version = 'b6dc27d953ff61913897fb0ec9be6f0236210f09'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
