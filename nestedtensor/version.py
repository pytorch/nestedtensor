__version__ = '0.0.1.dev202052017+787ba16'
git_version = '787ba16ac5d8294786706ae0d19c945b262585b9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
