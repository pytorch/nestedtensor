__version__ = '0.0.1.dev20209112+a06b9ce'
git_version = 'a06b9ceae821ba1d9e91ac6d7564a28cd6ae1add'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
