__version__ = '0.0.1+1956b4a'
git_version = '1956b4a2bf92125163adf945015c8c387e321ae2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
