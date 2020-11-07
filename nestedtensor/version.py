__version__ = '0.0.1.dev20201171+0d95817'
git_version = '0d958178d200c71fd5a799be555f8ae4d0e2a1a6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
