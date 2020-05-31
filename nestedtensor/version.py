__version__ = '0.0.1.dev202053121+54658b7'
git_version = '54658b780793162b7aab6eedbc26b8bdf96bf157'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
