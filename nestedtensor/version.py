__version__ = '0.0.1.dev2020647+1bea1f2'
git_version = '1bea1f295a669d84ad9190f12b3177cb92828c37'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
