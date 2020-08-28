__version__ = '0.0.1.dev20208282+47a73b6'
git_version = '47a73b6d6953fc97110e4cf01bcb6ae447923741'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
