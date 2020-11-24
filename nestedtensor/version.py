__version__ = '0.0.1.dev2020112416+bd9bb35'
git_version = 'bd9bb353f9b28013c04676138cf443f6a27a1e92'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
