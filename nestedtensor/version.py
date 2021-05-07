__version__ = '0.1.4+4f5ffe2'
git_version = '4f5ffe2289b9dc5aa0f04501a2f764adbcde08ea'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
