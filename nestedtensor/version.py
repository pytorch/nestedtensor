__version__ = '0.0.1.dev20202260+8f90718'
git_version = '8f9071809ac376d2e5405bcb35555e8d59c0ce4d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
