__version__ = '0.0.1.dev2020665+a570693'
git_version = 'a570693e6f935528f1086656e8c32b0cb6d022a2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
