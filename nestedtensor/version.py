__version__ = '0.0.1.dev2019121220+69be765'
git_version = '69be765876c910d634b0e30c5960031d6f2e40ac'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
