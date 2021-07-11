__version__ = '0.1.4+1fc28de'
git_version = '1fc28deca4c53569c7e43468617a3ea60e38d4fa'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
