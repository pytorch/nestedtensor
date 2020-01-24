__version__ = '0.0.1.dev202012421+7227581'
git_version = '722758177a90bd1a33d596f292a6c483469d8094'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
