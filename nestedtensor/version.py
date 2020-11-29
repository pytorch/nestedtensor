__version__ = '0.0.1.dev2020112923+0e6b698'
git_version = '0e6b698b836a4386fd04073e3d70378764b5f7a1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
