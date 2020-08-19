__version__ = '0.0.1.dev20208191+852d670'
git_version = '852d670f784f0fe0a34b772eb778954636318662'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
