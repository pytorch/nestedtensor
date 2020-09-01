__version__ = '0.0.1.dev20209122+02be12d'
git_version = '02be12d91cd33589184819319caba64c725b3e76'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
