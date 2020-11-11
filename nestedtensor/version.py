__version__ = '0.0.1.dev2020111120+98e8828'
git_version = '98e8828da19e477f3005b49fc9208a21882b6774'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
