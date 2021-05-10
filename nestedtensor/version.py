__version__ = '0.1.4+6bbe4e8'
git_version = '6bbe4e8216024b9876a4d8c5f4c7dea6adecca89'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
