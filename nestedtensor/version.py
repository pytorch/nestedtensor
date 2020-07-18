__version__ = '0.0.1.dev20207183+dcb7f4c'
git_version = 'dcb7f4c5dbbe2b53397a6eaabe2b590ca138b592'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
