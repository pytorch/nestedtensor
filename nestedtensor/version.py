__version__ = '0.0.1+981741d'
git_version = '981741dd224cee7d77a2359142d45b5b879ccbe2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
