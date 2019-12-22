__version__ = '0.0.1.dev201912225+37a0ea4'
git_version = '37a0ea4e4a1b26e6568f76283ce32fde41d57de0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
