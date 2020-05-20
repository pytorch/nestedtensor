__version__ = '0.0.1.dev20205202+25193df'
git_version = '25193dffc540b51755b1e01738802c11f6c4690e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
