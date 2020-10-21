__version__ = '0.0.1.dev2020102019+03c63c0'
git_version = '03c63c0d110b13b78eb3ea6d04895511b8b7026c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
