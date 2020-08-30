__version__ = '0.0.1.dev20208302+99458cc'
git_version = '99458ccef8078c9045a3574634eef98a4c5e1de1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
