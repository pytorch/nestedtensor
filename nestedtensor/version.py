__version__ = '0.0.1.dev202052020+b9fcbb0'
git_version = 'b9fcbb0bf8f205934a493d8bd308c69bd09c9d2e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
