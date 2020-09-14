__version__ = '0.0.1.dev202091421+3f93ad6'
git_version = '3f93ad66bb1a7b50dba457a9aaae00fbcff9f05e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
