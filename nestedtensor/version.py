__version__ = '0.0.1.dev20209421+f9297c7'
git_version = 'f9297c7e09c79ba10551d68c4f3caf5887911516'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
