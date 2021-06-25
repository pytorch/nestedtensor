__version__ = '0.1.4+4eae146'
git_version = '4eae146a30b6194ebf9a35c25344786f54030189'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
