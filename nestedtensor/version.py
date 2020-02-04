__version__ = '0.0.1.dev20202420+37bff65'
git_version = '37bff6570b87fee7669b3746866ce2e7ab4e5b74'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
