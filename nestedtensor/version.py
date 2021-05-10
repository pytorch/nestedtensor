__version__ = '0.1.4+c0994cd'
git_version = 'c0994cd83a15f612e4e4fc61093bb822d666c122'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
