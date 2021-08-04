__version__ = '0.1.4+c381487'
git_version = 'c3814879bc85a78008ececf59a19641d5acac833'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
