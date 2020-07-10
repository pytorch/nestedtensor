__version__ = '0.0.1.dev202071019+237990a'
git_version = '237990ab4945b4c4fcbd87abd9cbc85f9dcabaa0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
