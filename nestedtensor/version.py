__version__ = '0.1.4+efeee8e'
git_version = 'efeee8e6900b5b3f98d8debff1feafe3b5e0097c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
