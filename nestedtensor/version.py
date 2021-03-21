__version__ = '0.0.1+b76f151'
git_version = 'b76f151ee0e900094bab1af4393fedbab99b0d2b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
