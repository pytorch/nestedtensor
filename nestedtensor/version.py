__version__ = '0.0.1.dev20208205+bbe3e41'
git_version = 'bbe3e41989e3b86fb1a8aa0b4551f684ef2b7224'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
