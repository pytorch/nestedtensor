__version__ = '0.0.1.dev20208241+e0443ac'
git_version = 'e0443acb0fa030d5a5116d4cacb1800c6a964cd6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
