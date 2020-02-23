__version__ = '0.0.1.dev20202230+663969b'
git_version = '663969bf0b02b1bc1e1b5361d3fdc006a8cc8a32'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
