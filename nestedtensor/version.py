__version__ = '0.1.4+28501c3'
git_version = '28501c30732e7ce460ab47eeecec6057ecbc6989'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
