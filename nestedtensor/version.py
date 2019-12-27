__version__ = '0.0.1.dev2019122721+5f921db'
git_version = '5f921db935db2b44647f48c0b6fa736164190c0c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
