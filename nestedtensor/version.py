__version__ = '0.0.1.dev20206173+600c623'
git_version = '600c6239149b2b8123c01b633809b8e8348e4977'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
