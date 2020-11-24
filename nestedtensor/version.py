__version__ = '0.0.1.dev2020112420+022e356'
git_version = '022e35629482ccbf9a3ee6e214cb60e616ce820d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
