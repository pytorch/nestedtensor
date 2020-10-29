__version__ = '0.0.1.dev2020102921+32b6150'
git_version = '32b6150cc6ab239360068bb335dbc84cc61ab369'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
