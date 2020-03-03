__version__ = '0.0.1.dev20203323+69a9620'
git_version = '69a9620b44d0370020852c5e576cccd290f117bb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
