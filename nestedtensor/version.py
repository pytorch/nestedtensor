__version__ = '0.0.1.dev202010717+c464523'
git_version = 'c4645231b24dffd2863515e184ed7d29151399a5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
