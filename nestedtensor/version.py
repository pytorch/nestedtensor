__version__ = '0.0.1.dev202032617+2701a47'
git_version = '2701a47b4a897c2e34309b5b536e6cf3222d4d09'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
