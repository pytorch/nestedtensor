__version__ = '0.0.1.dev202082023+b17751e'
git_version = 'b17751ec125038d0af4afa100e61088d730b7fb8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
