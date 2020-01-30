__version__ = '0.0.1.dev20201306+60112ae'
git_version = '60112aea90cc54c98e5997bee9166ef94ee0e114'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
