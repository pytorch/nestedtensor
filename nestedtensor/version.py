__version__ = '0.0.1.dev201912282+a2ca671'
git_version = 'a2ca671ba806919c8744dd944e6f34e2ec4bf1c0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
