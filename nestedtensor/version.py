__version__ = '0.0.1.dev20203291+92278df'
git_version = '92278df843d58e5e4414360babc7352134bcc824'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
