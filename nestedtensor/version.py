__version__ = '0.1.4+243d7b5'
git_version = '243d7b503e56244c2a4af524e960785ad3a974ef'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
