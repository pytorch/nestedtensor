__version__ = '0.0.1.dev20206173+e50c390'
git_version = 'e50c390cef79aebbb292fd90b7d44e04e4274b87'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
