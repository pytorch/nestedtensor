__version__ = '0.0.1.dev201912916+0e4233e'
git_version = '0e4233e4a9004ed07a8c489a88da38330b8017b8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
