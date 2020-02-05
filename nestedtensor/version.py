__version__ = '0.0.1.dev2020254+b7f4a5c'
git_version = 'b7f4a5c1e2e9bc69ff1436f4be6c5dbb43d2e69d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
