__version__ = '0.0.1.dev202032621+b220535'
git_version = 'b220535d6a3d21876cb895cb684537e4a66106b6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
