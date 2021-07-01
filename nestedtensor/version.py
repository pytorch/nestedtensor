__version__ = '0.1.4+4eb3555'
git_version = '4eb3555619de93c87d0ec2af62e715d3a2961273'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
