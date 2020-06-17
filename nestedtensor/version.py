__version__ = '0.0.1.dev20206172+d83f2d9'
git_version = 'd83f2d9961f77abb183a9bcc7ec63d246bf9d48d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
