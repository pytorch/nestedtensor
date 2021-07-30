__version__ = '0.1.4+66764fd'
git_version = '66764fd10e9b6f9c0710840d0cb17369b9d994be'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
