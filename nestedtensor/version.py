__version__ = '0.1.4+85de459'
git_version = '85de4590f4bbba84f5335b102f579ade7c72c8cf'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
