__version__ = '0.1.4+8cf536f'
git_version = '8cf536f99e846dc9e334f4abe955e671f5563bf7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
