__version__ = '0.0.1.dev20203322+8343363'
git_version = '8343363a8c7cab1e4fe9231ec2d5811f55dbbbc3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
