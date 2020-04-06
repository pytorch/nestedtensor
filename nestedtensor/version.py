__version__ = '0.0.1.dev20204620+fb11447'
git_version = 'fb1144700d8a19532837f467b8d66c6fbf21ded2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
