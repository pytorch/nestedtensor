__version__ = '0.0.1.dev202061717+8af11c0'
git_version = '8af11c0f3966f8221f09960ca91f589ac398bc47'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
