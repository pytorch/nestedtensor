__version__ = '0.0.1.dev201911722+cc3caf6'
git_version = 'cc3caf68587175fee7084140538ef6fe933a45cc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
