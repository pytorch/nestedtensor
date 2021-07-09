__version__ = '0.1.4+01e8112'
git_version = '01e81127dfb83278cc01d4fd0dad4713857ec6c8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
