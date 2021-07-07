__version__ = '0.1.4+18ab67e'
git_version = '18ab67e3cbcc4966bfd9304274ba2853158c0448'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
