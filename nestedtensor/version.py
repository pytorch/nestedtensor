__version__ = '0.1.4+592e713'
git_version = '592e713422be89ebd7abd4aac418ab4cee0b9aad'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
