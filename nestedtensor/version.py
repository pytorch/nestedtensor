__version__ = '0.0.1.dev20204252+63ec6f3'
git_version = '63ec6f3cfc4a1fa07ede2c11eaf1b73b340188f0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
