__version__ = '0.1.4+ab0b7bb'
git_version = 'ab0b7bb8fa5048c295dcdf7b81fc6b0edacf5378'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
