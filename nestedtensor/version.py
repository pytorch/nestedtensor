__version__ = '0.0.1.dev20203242+a9b19a9'
git_version = 'a9b19a95709dc482d654bc1fb7326b90cd4dd2d9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
