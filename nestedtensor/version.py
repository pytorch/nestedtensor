__version__ = '0.0.1.dev202022119+0d5bed2'
git_version = '0d5bed222308ef639ff2f2d2d9ea8315127ddcb6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
