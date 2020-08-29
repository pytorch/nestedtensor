__version__ = '0.0.1.dev20208292+2e15e8c'
git_version = '2e15e8c92259b33b0837c201040bf9379b30d4f5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
