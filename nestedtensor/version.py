__version__ = '0.1.4+655b65e'
git_version = '655b65eafe58df251eff8215ee7e01fe4bbe2b89'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
