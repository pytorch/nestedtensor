__version__ = '0.0.1.dev2020665+dc48880'
git_version = 'dc4888087b979c70b46b7e2a158c420c2b7c8b75'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
