__version__ = '0.0.1.dev20202418+4a756c8'
git_version = '4a756c8398d128882702d023fc09a57558431427'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
