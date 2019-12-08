__version__ = '0.0.1.dev20191281+274770c'
git_version = '274770c7eeb729557ca267fd4ab3a142c98af985'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
