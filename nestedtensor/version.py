__version__ = '0.0.1.dev202011260+1f3451e'
git_version = '1f3451e23a89a235a7e749b727dff5f9186caf1e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
