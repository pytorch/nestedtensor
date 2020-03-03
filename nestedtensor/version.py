__version__ = '0.0.1.dev20203322+e6ed593'
git_version = 'e6ed59353c3f28471c9c302f5f642519dbc74921'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
