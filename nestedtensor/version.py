__version__ = '0.1.4+005f39c'
git_version = '005f39c7452a860e47604b3b6a2ba4d68253fef1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
