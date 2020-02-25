__version__ = '0.0.1.dev20202253+6f527f5'
git_version = '6f527f58234f579385a43e5f9068a5d216cdd23d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
