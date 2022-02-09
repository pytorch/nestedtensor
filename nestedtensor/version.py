__version__ = '0.1.4+6169905'
git_version = '61699056144c04273dc0dc702253d4910ffcbd5b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
