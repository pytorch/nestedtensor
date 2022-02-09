__version__ = '0.1.4+b7c17cb'
git_version = 'b7c17cb1c0457faa97eafd8693c3589cec877520'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
