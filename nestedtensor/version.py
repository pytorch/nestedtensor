__version__ = '0.0.1.dev2020112216+a19c676'
git_version = 'a19c6762b9f2b353b9289ec8f8729e10605ff563'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
