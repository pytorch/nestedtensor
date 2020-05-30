__version__ = '0.0.1.dev202053021+4dfd2b6'
git_version = '4dfd2b648a1e53c8df9294015f9add5150f0c7bc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
