__version__ = '0.0.1.dev202013019+5bae24a'
git_version = '5bae24a6c8a295af4931710284bce0b418deb390'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
