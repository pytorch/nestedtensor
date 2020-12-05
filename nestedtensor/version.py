__version__ = '0.0.1+ce7acb4'
git_version = 'ce7acb48b5006e8dc3bd5b96c406c0f709087aa8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
