__version__ = '0.0.1.dev2019121323+70c2ef0'
git_version = '70c2ef0738fecde903a1f65fcf6239a1fe6d8e0a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
