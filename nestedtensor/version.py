__version__ = '0.1.4+10ff983'
git_version = '10ff983e8d8887ff1cba83e3b25622f2fc7b1295'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
