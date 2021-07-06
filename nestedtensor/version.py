__version__ = '0.1.4+9573c98'
git_version = '9573c980aa1174c98b1f287335f311d7a049b351'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
