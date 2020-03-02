__version__ = '0.0.1.dev20203215+75f3647'
git_version = '75f36479438cd4b2a977bf9c3ecfebbea1924502'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
