__version__ = '0.0.1.dev20203415+25a00f5'
git_version = '25a00f5d099ec493afe4257380e8f21ad1ee5681'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
