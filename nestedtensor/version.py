__version__ = '0.0.1.dev20206172+0368e47'
git_version = '0368e472bcde8282525b12b4e5b50ac3016a6786'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
