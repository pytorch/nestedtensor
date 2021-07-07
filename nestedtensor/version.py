__version__ = '0.1.4+11137b5'
git_version = '11137b5532cb43394e5b98ec1f02e13ccfa89641'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
