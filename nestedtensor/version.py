__version__ = '0.0.1.dev202053019+83e4ab0'
git_version = '83e4ab06581c90ea61466a687a7ea13e1a0751bf'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
