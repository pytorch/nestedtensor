__version__ = '0.0.1.dev20208281+d08afe4'
git_version = 'd08afe49ed9da6536b1dd470e2158ef91054e5cb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
