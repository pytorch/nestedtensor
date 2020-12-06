__version__ = '0.0.1+952558e'
git_version = '952558ef0cc0d2f12f69d0f59ffe84a9cf5ecb10'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
