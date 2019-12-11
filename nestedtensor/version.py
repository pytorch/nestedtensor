__version__ = '0.0.1.dev2019121119+247bca3'
git_version = '247bca346ab37a4286881a7ba834721a5615118a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
