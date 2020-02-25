__version__ = '0.0.1.dev20202254+1fbc44b'
git_version = '1fbc44b3fdbdb07aaafa426ebf041b56e311a144'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
