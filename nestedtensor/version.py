__version__ = '0.1.4+95ac774'
git_version = '95ac774435b94c95072f242a3ea423d538f404e3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
