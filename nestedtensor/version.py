__version__ = '0.1.4+e080eb6'
git_version = 'e080eb659e4ab0a3559d9b1e057874215c3dc941'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
