__version__ = '0.1.4+c78d04f'
git_version = 'c78d04fb547079dc9e7efda6fede573acd4a9996'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
