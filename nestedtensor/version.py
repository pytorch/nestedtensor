__version__ = '0.0.1.dev20209160+337242c'
git_version = '337242ce0a4574022fc45d443e320d118fc9ffa8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
