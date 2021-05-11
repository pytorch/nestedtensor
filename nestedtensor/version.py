__version__ = '0.1.4+982a5f5'
git_version = '982a5f50c14fd8116f86211ffb6bebd73e00197f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
