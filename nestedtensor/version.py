__version__ = '0.0.1.dev20203221+f5de2ee'
git_version = 'f5de2ee3576e334b39e42e98008accf9e275f4de'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
