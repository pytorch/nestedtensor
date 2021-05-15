__version__ = '0.1.4+b5f8441'
git_version = 'b5f84416a15277251f26e2fef4178cf56fdb5cc1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
