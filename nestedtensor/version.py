__version__ = '0.1.4+31fbd42'
git_version = '31fbd424487e4edcf38fc5a76fa31808515e990d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
