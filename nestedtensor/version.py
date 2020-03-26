__version__ = '0.0.1.dev202032623+b75e357'
git_version = 'b75e357a7278fa493912dbd39774f1d7c65c97ab'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
