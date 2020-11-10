__version__ = '0.0.1.dev2020111021+24846ff'
git_version = '24846ffc10512e5a59a61134980087bc07fa5139'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
