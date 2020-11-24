__version__ = '0.0.1.dev2020112416+256cc96'
git_version = '256cc96f9e03027797babde6e3ebd26928c03ce8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
