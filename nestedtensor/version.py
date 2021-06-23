__version__ = '0.1.4+fe665be'
git_version = 'fe665be78daa361df1ba4840c0efa8d87a716b05'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
