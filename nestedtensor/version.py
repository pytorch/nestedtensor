__version__ = '0.0.1.dev2019121120+d716096'
git_version = 'd716096e520fb92915f5fb77db7aaad07181c934'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
