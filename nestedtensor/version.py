__version__ = '0.0.1.dev202032921+bc62b71'
git_version = 'bc62b7119bc0fb29511cf54699b69ce9724f02bb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
