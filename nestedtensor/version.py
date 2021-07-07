__version__ = '0.1.4+cc4d5a4'
git_version = 'cc4d5a43b7c180cb6ad347c7d4c92e851bd02744'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
