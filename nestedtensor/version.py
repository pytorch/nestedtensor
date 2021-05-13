__version__ = '0.1.4+3f9cba8'
git_version = '3f9cba8007540ab173d4c17f18330e4e347ac38a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
