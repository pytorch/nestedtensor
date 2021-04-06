__version__ = '0.0.1+7ce33ea'
git_version = '7ce33ea45a02faa3bfeac81ec472135a53d772db'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
