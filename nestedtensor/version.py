__version__ = '0.0.1.dev20209421+b677b4b'
git_version = 'b677b4b0c10033891d20a90813c44fa84a8d808b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
