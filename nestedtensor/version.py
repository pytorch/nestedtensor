__version__ = '0.0.1.dev20207153+063d303'
git_version = '063d303499da20e0b0c9373cf5831f263dc672ca'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
