__version__ = '0.0.1.dev2020665+66bca05'
git_version = '66bca053a2545709e031caf6373bbe630fd75dec'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
