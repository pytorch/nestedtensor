__version__ = '0.0.1.dev20207193+3e4a044'
git_version = '3e4a0446ed4375d30747cb001428f276440a584d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
