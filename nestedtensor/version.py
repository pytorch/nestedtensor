__version__ = '0.1.4+9f29fb4'
git_version = '9f29fb45f4c7c1affd1c7a0dd1947de613e1e908'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
