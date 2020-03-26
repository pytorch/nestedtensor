__version__ = '0.0.1.dev20203264+0bd5db9'
git_version = '0bd5db99c85e83c17af164ce90af3321afe624da'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
