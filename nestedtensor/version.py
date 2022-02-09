__version__ = '0.1.4+a991f73'
git_version = 'a991f739eba015d34980027440b53c14a207b054'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
