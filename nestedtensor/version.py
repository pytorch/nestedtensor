__version__ = '0.0.1.dev2020112522+33ba06a'
git_version = '33ba06a2093e55e38dccf20b7e00aac2c01cdff4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
