__version__ = '0.0.1.dev2020271+642d00c'
git_version = '642d00ce49b768ac95a8c3285adc974f21c0482b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
