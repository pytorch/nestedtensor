__version__ = '0.1.4+9f84211'
git_version = '9f8421150b5838c7fb28109611459b12ab315968'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
