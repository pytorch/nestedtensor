__version__ = '0.0.1.dev20208417+dfa070e'
git_version = 'dfa070ed531b46211c2f24d7da910e81179358a6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
