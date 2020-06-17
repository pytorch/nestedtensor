__version__ = '0.0.1.dev20206172+91f4374'
git_version = '91f4374ace7649c63ae5e33af1f0aa092e78400f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
