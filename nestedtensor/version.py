__version__ = '0.0.1.dev2020112023+5d422c5'
git_version = '5d422c501223b5a83322907ef0e6ed49c56bfe30'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
