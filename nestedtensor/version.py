__version__ = '0.0.1.dev20201122+f439e4e'
git_version = 'f439e4eac6ff826b7c9951ae45706863be0dc82a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
