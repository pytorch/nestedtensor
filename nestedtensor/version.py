__version__ = '0.0.1.dev20202174+d657520'
git_version = 'd6575206619447fa5b407238ab46963aa969da2d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
