__version__ = '0.0.1.dev202053019+1e3e640'
git_version = '1e3e640b009c13184ba2a18357a72d018a27d210'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
