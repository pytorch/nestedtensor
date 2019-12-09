__version__ = '0.0.1.dev201912921+3c811e0'
git_version = '3c811e09300b3d02bbe910f54622eb374cffb3fc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
