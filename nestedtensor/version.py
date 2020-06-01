__version__ = '0.0.1.dev2020613+754ddea'
git_version = '754ddea51035efa51f38699bedcc88f036bc597c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
