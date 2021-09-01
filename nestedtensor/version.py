__version__ = '0.1.4+29d58d8'
git_version = '29d58d85ccd2c1b47fc40f2786f08c713f49acc6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
