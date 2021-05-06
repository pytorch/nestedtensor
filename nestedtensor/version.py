__version__ = '0.1.4+2569165'
git_version = '2569165eea04c66b658f35c3f374123ecb7d7226'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
