__version__ = '0.0.1.dev202022019+f811029'
git_version = 'f811029fa27c1effa926d5dce4cf10a4eb8f4b0b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
