__version__ = '0.0.1.dev2020111020+f522f19'
git_version = 'f522f199f0b830142c121d42bbb46aae9fa323c9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
