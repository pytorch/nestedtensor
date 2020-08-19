__version__ = '0.0.1.dev20208190+8065e72'
git_version = '8065e72678898a6490cb516bb766fe0b9a972870'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
