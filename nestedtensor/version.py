__version__ = '0.0.1.dev20208304+ba0ff6e'
git_version = 'ba0ff6e5f500b8cf86fc08423aaa9ac418d4b92f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
