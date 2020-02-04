__version__ = '0.0.1.dev20202420+823d924'
git_version = '823d9245866ec8deac3557a3e24685ddbd200a65'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
