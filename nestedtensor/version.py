__version__ = '0.1.4+c4661e9'
git_version = 'c4661e941ad9b54c3acf48e3876a3f061c51abe6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
