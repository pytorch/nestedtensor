__version__ = '0.0.1.dev20202172+53147be'
git_version = '53147be770c28cdea621592945af684b766c89fd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
