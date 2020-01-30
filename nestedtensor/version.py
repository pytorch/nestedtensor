__version__ = '0.0.1.dev20201301+6e50874'
git_version = '6e508744c58aa1bb98035ad0c605ebd2cfc38f50'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
