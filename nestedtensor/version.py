__version__ = '0.0.1.dev202013022+5c7d43a'
git_version = '5c7d43ac42a6a4a7e6bec9d027603caf4b456eb8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
