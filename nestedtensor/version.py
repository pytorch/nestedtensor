__version__ = '0.1.4+f64f01e'
git_version = 'f64f01e7dba486f1a8622b5e38257529ee9e905c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
