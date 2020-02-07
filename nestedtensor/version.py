__version__ = '0.0.1.dev20202721+ee8b024'
git_version = 'ee8b024d3c7d952a7573b1aa24b5e1071c6e3e08'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
