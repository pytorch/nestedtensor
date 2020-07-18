__version__ = '0.0.1.dev20207183+ebd4e54'
git_version = 'ebd4e5453a257f7696018c4dff5c3a19276980fa'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
