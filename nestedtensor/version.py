__version__ = '0.1.4+affab42'
git_version = 'affab42088d142b0f55beedda5f4404444fd2abc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
