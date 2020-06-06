__version__ = '0.0.1.dev2020664+ddc0889'
git_version = 'ddc08894939441dac802dfbd049cda8bab54422e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
