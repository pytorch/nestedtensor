__version__ = '0.0.1.dev20203215+669a17b'
git_version = '669a17bd6a6cfbfd76d97b11fb2499a0f539b136'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
