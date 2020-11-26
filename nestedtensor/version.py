__version__ = '0.0.1.dev2020112616+6fae2bf'
git_version = '6fae2bf39f50995b986fcb38206f0096394d9dee'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
