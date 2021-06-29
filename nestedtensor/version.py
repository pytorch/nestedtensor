__version__ = '0.1.4+036a159'
git_version = '036a15938527fbf84805e96c9679cf7e4d432b0c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
