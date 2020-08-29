__version__ = '0.0.1.dev20208290+960b82a'
git_version = '960b82aca518c4984fbd6f5229bc9998ba26f9a4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
