__version__ = '0.0.1.dev202032323+366c5fb'
git_version = '366c5fb75ba84eb937fdbd400454351a19da02c6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
