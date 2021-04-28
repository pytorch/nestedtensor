__version__ = '0.0.1+f79ec60'
git_version = 'f79ec60c02c783defbc0b3f586eefa721496fa07'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
