__version__ = '0.0.1.dev20204915+150aeaa'
git_version = '150aeaa7106ec2bd9037f40a571a36e2073d31d3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
