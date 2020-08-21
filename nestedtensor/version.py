__version__ = '0.0.1.dev202082116+b668470'
git_version = 'b668470c3f6cf1b02c5d78a92d9824b60dec2848'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
