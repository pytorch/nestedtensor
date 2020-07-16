__version__ = '0.0.1.dev202071621+5953886'
git_version = '59538868e112f33c437ca0ac3c9ec8dd219803f9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
