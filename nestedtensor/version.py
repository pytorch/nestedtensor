__version__ = '0.0.1.dev201912722+dfde953'
git_version = 'dfde953205f6fbc3fc01706a6ce16963fbc81a4a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
