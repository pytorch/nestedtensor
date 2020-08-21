__version__ = '0.0.1.dev20208216+dc014c0'
git_version = 'dc014c0547d45303ea8d58d2f78b36bfce3da750'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
