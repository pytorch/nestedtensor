__version__ = '0.0.1.dev20206105+313dbce'
git_version = '313dbcec5900dfbf40f0c849acd9e7608cdc6ee2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
