__version__ = '0.0.1.dev202071520+a4417d4'
git_version = 'a4417d4720708bcde9eac44d4866015b8004205c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
