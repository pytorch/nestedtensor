__version__ = '0.0.1+aded2ac'
git_version = 'aded2acc0344d2115e35661614192dadcfa94fe6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
