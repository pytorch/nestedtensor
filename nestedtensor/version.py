__version__ = '0.0.1+9a3ff0c'
git_version = '9a3ff0c88624b88f069510fcd56047f5661f0130'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
