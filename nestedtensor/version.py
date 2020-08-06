__version__ = '0.0.1.dev20208617+4561ec9'
git_version = '4561ec9d887654d86ba94232b32425b40cf774bf'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
