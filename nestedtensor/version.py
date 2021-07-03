__version__ = '0.1.4+b16cb93'
git_version = 'b16cb93e38a4a1abaced02eb96ef657eba008e96'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
