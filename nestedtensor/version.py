__version__ = '0.0.1.dev201912193+ba84c9c'
git_version = 'ba84c9cc4075fe00fd151295e1ea4b3e9251177c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
