__version__ = '0.0.1.dev202011623+dacd0a7'
git_version = 'dacd0a7376d296f862ecc880dcbdd4b1eaf76883'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
