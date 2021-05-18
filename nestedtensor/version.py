__version__ = '0.1.4+581ade8'
git_version = '581ade8106ef9eda94da2983df8f586fe3ee7608'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
