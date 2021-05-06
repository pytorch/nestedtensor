__version__ = '0.1.4+2027399'
git_version = '2027399cacdc29326c81383fc80b2611ee8b6e4a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
