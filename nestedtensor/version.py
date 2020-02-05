__version__ = '0.0.1.dev2020252+b259069'
git_version = 'b2590695a352bd89b88353f34eece6a158a8db03'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
