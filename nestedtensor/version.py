__version__ = '0.0.1.dev20201206+342b1b2'
git_version = '342b1b253609270c7e84d09e8c874c9dab701d64'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
