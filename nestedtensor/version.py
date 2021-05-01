__version__ = '0.1.4+2f73b45'
git_version = '2f73b45142c324b6e4ba8fc4e72a11eb4c577e30'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
