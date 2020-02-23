__version__ = '0.0.1.dev20202237+a2102d4'
git_version = 'a2102d4589a856aa3dc6d1b51cc98ecdc94908fe'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
