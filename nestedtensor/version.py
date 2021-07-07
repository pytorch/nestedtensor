__version__ = '0.1.4+ed44c90'
git_version = 'ed44c90477316e4878f05761fc062f0fde135ad3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
