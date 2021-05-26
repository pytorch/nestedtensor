__version__ = '0.1.4+4ba9cfa'
git_version = '4ba9cfa497986b47f74703af68136955afbd02a9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
