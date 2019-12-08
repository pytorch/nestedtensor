__version__ = '0.0.1.dev20191281+9088aa2'
git_version = '9088aa25154bfd0872273bbb35637a51afc77964'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
