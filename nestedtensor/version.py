__version__ = '0.0.1.dev202013118+a867298'
git_version = 'a86729832da120021e2d44d5ac8d537ab85b94ea'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
