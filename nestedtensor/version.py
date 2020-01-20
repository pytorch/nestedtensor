__version__ = '0.0.1.dev20201201+5d427e1'
git_version = '5d427e17d236803647fe3d0c04c72235610aaf24'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
