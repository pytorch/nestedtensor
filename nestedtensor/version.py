__version__ = '0.0.1.dev20202719+cd2a562'
git_version = 'cd2a562efe00aae1b9f40e7cd30aac46df5c718d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
