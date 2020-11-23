__version__ = '0.0.1.dev2020112319+cab759f'
git_version = 'cab759fc8066382f422bab331ad9494a6794fd84'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
