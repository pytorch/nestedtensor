__version__ = '0.0.1.dev2019122717+774588f'
git_version = '774588f1c41560a21a3aa4bf7620ba817da1948b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
