__version__ = '0.1.4+bb5420c'
git_version = 'bb5420cfd0782fd22a0319ba2df4360533a8bd2a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
