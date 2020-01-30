__version__ = '0.0.1.dev20201302+618474b'
git_version = '618474b529c8dbc0fc38b0835918db3d5d0e76fa'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
