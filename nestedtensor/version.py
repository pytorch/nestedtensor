__version__ = '0.0.1.dev20205194+d5303b1'
git_version = 'd5303b17c80c998aec290a57cb2f59ad0ec06c70'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
