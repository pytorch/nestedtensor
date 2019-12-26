__version__ = '0.0.1.dev2019122422+ffa374e'
git_version = 'ffa374ef98019505f681206b389f71170df11690'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
