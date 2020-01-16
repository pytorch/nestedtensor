__version__ = '0.0.1.dev202011621+eb44215'
git_version = 'eb44215521b7552dbf78de760e4684e01cf88719'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
