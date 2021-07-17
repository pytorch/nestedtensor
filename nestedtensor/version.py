__version__ = '0.1.4+c4a5147'
git_version = 'c4a514795d2fa209f8df64754410b5262b3dfa45'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
