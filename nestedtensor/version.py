__version__ = '0.1.4+34337d0'
git_version = '34337d07cdb4d4ba5198024ea5df34d20f055dab'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
