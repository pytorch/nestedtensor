__version__ = '0.0.1.dev20202423+c676cdf'
git_version = 'c676cdfdbcfb71991afbf1c4c34dbd518c3ee95a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
