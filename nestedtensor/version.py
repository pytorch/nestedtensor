__version__ = '0.1.4+cc22d92'
git_version = 'cc22d92973a1fd0af3ac21ad15baa964cdefdc16'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
