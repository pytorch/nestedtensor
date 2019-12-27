__version__ = '0.0.1.dev201912270+8754a73'
git_version = '8754a73d85e8b24c1783a722cfdb43b1ddbc1b22'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
