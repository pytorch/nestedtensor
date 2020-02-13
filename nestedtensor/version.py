__version__ = '0.0.1.dev20202132+898e22d'
git_version = '898e22d48159006a489bbce0860de275013f6392'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
