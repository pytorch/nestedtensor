__version__ = '0.0.1+ccc2f9b'
git_version = 'ccc2f9b8721f55bb6e9fedc3676519dd8d4c7745'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
