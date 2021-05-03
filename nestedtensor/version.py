__version__ = '0.1.4+85a7812'
git_version = '85a781211d27ef133980de923f053b6efaa2ea3b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
