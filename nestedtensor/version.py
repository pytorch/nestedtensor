__version__ = '0.0.1.dev20208153+8791764'
git_version = '87917640875dbc7b36595f826e6e517f343c76a2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
