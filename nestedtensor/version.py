__version__ = '0.1.4+b4af654'
git_version = 'b4af6546cc11bf8458a0c69fb09d92f961eab014'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
