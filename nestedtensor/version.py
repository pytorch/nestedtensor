__version__ = '0.1.4+bb0f72e'
git_version = 'bb0f72e126f8661b364977841c6bbaa6e04654aa'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
