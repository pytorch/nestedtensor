__version__ = '0.0.1.dev20206174+b68bb26'
git_version = 'b68bb26f769cbb75424d1915724f6c09dab9016b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
