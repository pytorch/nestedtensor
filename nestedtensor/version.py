__version__ = '0.0.1.dev20208113+6d1ca2c'
git_version = '6d1ca2cee863adc306d9948dfed2ccf2bdb38b98'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
