__version__ = '0.1.4+4b45eda'
git_version = '4b45eda3145f5c3a3452d759660cf12c055007f1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
