__version__ = '0.0.1.dev20208271+3bb5d8d'
git_version = '3bb5d8d6e2b4517b595fcdfc8cba01295976ecf9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
