__version__ = '0.0.1+29396ee'
git_version = '29396ee4daeb33cfef3d04348a6b80b700773421'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
