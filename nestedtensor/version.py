__version__ = '0.0.1.dev20206518+72035a4'
git_version = '72035a49bb96d186e679303c2b12496b33c16c5d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
