__version__ = '0.1.4+99b3d96'
git_version = '99b3d96b3876bdbc80ad68e50a1d37ebb44eeb66'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
