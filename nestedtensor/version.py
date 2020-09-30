__version__ = '0.0.1.dev202093014+0c940eb'
git_version = '0c940eb65c059ead15266ba46a8fa45b4e162218'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
