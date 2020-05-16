__version__ = '0.0.1.dev20205162+51d3be0'
git_version = '51d3be07feac2c500b1a73a550fb668560573ec3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
