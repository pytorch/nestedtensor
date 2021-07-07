__version__ = '0.1.4+3d58ae6'
git_version = '3d58ae611c888fa6673d7e5123c58bb7d4faed6d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
