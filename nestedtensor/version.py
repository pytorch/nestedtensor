__version__ = '0.0.1.dev202012317+1349ce4'
git_version = '1349ce48bec97938041d12aa328db90067313d32'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
