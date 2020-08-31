__version__ = '0.0.1.dev20208315+d1c0ecc'
git_version = 'd1c0ecc9108c9b8de1735e8d38f2fe70264f7a57'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
