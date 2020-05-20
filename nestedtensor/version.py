__version__ = '0.0.1.dev202052020+f671d78'
git_version = 'f671d78b7291e2949b3cd2f924ebd7f54baf87c0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
