__version__ = '0.1.4+83cefbb'
git_version = '83cefbbed4431344fd69872751c52c6d18f6390c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
