__version__ = '0.0.1.dev201912264+8911706'
git_version = '8911706629b5b310b60d8cace987c75c8d138ba5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
