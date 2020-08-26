__version__ = '0.0.1.dev202082621+c66a81c'
git_version = 'c66a81c7f191fd8d5ad48e68a5a5a962208e5b85'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
