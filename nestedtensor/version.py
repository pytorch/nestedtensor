__version__ = '0.0.1.dev202093016+681f816'
git_version = '681f816e91588a98adcc13f167125b2e6be3f84b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
