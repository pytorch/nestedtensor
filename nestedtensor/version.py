__version__ = '0.0.1.dev20202290+7480b4f'
git_version = '7480b4f3fdcdd0a96d47c515430cdff1ed1d98d4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
