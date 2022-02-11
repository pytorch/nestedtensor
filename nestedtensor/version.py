__version__ = '0.1.4+93f0ada'
git_version = '93f0ada5d53a09991dd407b0a5816367bf2a4806'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
