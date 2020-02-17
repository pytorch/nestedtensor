__version__ = '0.0.1.dev202021722+1d33c5c'
git_version = '1d33c5c9c338fd4ffbad1809b5131b006a811005'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
