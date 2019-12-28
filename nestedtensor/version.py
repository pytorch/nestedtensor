__version__ = '0.0.1.dev201912286+fd70e32'
git_version = 'fd70e32e6a5e8f4c2ec5029c780cb1729e4551cd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
