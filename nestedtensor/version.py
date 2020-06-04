__version__ = '0.0.1.dev2020647+d88f565'
git_version = 'd88f565b3fcc88d4877513be6ea0a7a0fda22a7f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
