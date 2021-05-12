__version__ = '0.1.4+5a4c6e2'
git_version = '5a4c6e245b19a945de477db6eb31280c77dc0c8c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
