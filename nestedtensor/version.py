__version__ = '0.0.1.dev20203253+4e4cbf8'
git_version = '4e4cbf843ea7ab0effeaaa8d6caaa45f749fc443'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
