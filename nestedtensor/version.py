__version__ = '0.0.1.dev20202250+5e721e7'
git_version = '5e721e7b7488a725570598031ea3c35063040924'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
