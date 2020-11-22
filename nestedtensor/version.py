__version__ = '0.0.1.dev2020112216+6a88130'
git_version = '6a881300d80b33c1ef35e9e64c1d45297affc555'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
