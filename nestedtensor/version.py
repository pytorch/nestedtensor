__version__ = '0.0.1.dev20205301+78aab4e'
git_version = '78aab4e2ac7dcdb817611183f4c522b023ce6398'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
