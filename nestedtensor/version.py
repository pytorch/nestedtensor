__version__ = '0.1.4+5e0fa90'
git_version = '5e0fa90635cf65ed050d5f238237a048d422df4c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
