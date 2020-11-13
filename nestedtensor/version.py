__version__ = '0.0.1.dev202011131+58d1021'
git_version = '58d10214eb9f073a550220c48d6cf091fb241216'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
