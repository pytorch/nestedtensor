__version__ = '0.0.1.dev20209233+963ab31'
git_version = '963ab31e7309161044844df008d1e8f623853216'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
