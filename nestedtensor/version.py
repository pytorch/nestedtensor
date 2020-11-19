__version__ = '0.0.1.dev2020111922+a0e2d3e'
git_version = 'a0e2d3e0f76cd203c5484340a0d6a3e424e720dd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
