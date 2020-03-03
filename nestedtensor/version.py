__version__ = '0.0.1.dev20203322+02c25c3'
git_version = '02c25c30c2044bc5b907f5c02baea83a835da2c9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
