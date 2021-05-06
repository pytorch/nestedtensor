__version__ = '0.1.4+6edb11a'
git_version = '6edb11aaa78ea4d9a4e93f74a5166ba681cabc8e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
