__version__ = '0.1.4+e60f900'
git_version = 'e60f90010cd7864bd11c14f7f29a7449619c1ce5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
