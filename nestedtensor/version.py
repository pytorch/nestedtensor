__version__ = '0.1.4+c440806'
git_version = 'c44080652547e8d5f85a6f2e76fb14aefcca0b3f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
