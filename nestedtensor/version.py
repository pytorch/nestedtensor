__version__ = '0.0.1.dev2020111316+9f723a9'
git_version = '9f723a9b9f24e72d268ae22730eeff282965d36b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
