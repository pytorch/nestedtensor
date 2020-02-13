__version__ = '0.0.1.dev20202132+f1fd8fa'
git_version = 'f1fd8fac0c7ac80039cc5dcb17dd5795c4df6447'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
