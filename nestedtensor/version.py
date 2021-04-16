__version__ = '0.0.1+06b8787'
git_version = '06b87878eb228373229ddb5cfae14ed6bd0ec5df'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
