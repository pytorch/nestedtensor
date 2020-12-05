__version__ = '0.0.1+278318d'
git_version = '278318d63e62c0ccd0ff37a4c75b46baca1f6343'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
