__version__ = '0.0.1.dev202013016+9bb2a43'
git_version = '9bb2a4381f5ed33ec94955dd89330dfdf9e545ce'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
