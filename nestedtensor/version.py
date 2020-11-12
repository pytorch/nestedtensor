__version__ = '0.0.1.dev2020111218+320f217'
git_version = '320f217fb81f8a60869595a338c16dbd7dcc4ddf'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
