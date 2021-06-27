__version__ = '0.1.4+76ae09d'
git_version = '76ae09dff39934abb68c4ac72bcee86612fd837b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
