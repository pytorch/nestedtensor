__version__ = '0.0.1.dev20208714+3747381'
git_version = '37473811913acec8ac6a4693dc41311a2e65bce8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
