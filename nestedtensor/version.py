__version__ = '0.1.4+9445ae0'
git_version = '9445ae07fe4b7a590525ae2b2e6556102c1605f2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
