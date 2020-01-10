__version__ = '0.0.1.dev20201102+b980a10'
git_version = 'b980a10fe92c5d2619d1c7572fc59e008e731b83'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
