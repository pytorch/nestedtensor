__version__ = '0.0.1.dev20201301+4a75e70'
git_version = '4a75e7042194c0aeb688ef5529975baef3b230b9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
