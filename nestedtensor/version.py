__version__ = '0.1.4+e8df566'
git_version = 'e8df566da115fe9fe11f431493496ac24672cc9a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
