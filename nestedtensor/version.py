__version__ = '0.0.1.dev2019122723+a1305f1'
git_version = 'a1305f1706cef8c32d52ca02c026e644c538f493'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
