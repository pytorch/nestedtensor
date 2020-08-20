__version__ = '0.0.1.dev202082019+dc64b73'
git_version = 'dc64b7399f852fd32f6c5b551106d0cec9d92c28'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
