__version__ = '0.1.4+cd4fb75'
git_version = 'cd4fb751d9e787b9a3d65fdad6a4b62279213339'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
