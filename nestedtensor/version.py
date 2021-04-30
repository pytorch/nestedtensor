__version__ = '0.1.4+6ff3f3d'
git_version = '6ff3f3d7f6ab2d1376acf645b2b1f02d9aa1dd8b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
