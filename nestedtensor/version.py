__version__ = '0.1.4+13c39ed'
git_version = '13c39edd83a45d83dfaf4933381905c107d5a6aa'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
